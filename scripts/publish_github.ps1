param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')][string]$Repository,
    [string]$AssetDirectory = (Join-Path $PSScriptRoot '..\..\release-assets'),
    [switch]$SkipData
)
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'Install the official GitHub CLI first: https://cli.github.com/' }
& gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Sign in first with gh auth login, then run this script again.' }
Push-Location -LiteralPath $repoPath
try {
    $repositoryUrl = 'https://github.com/' + $Repository
    $config = 'window.PROJECT_CONFIG=' + (@{ repository=$repositoryUrl; releaseTag='v0.1' } | ConvertTo-Json -Compress) + ';'
    [IO.File]::WriteAllText((Join-Path $repoPath 'docs\assets\config.js'),$config,[Text.UTF8Encoding]::new($false))
    if (-not (Test-Path -LiteralPath (Join-Path $repoPath '.git'))) {
        & git init -b main
        if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
    }
    & git add .
    if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        & git commit -m 'Prepare open-set cross-corpus SER benchmark source, results and project page'
        if ($LASTEXITCODE -ne 0) { throw 'Configure your Git author identity and retry the commit.' }
    }
    & gh repo view $Repository --json name 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & gh repo create $Repository --public --source . --remote origin --push --description 'Open-Set Cross-Corpus Speech Emotion Recognition: benchmark code, features and paper results'
        if ($LASTEXITCODE -ne 0) { throw 'Repository creation/push failed' }
    } else {
        $existing = & git remote get-url origin 2>$null
        if ($LASTEXITCODE -eq 0 -and $existing -notin @($repositoryUrl,$repositoryUrl+'.git','git@github.com:'+$Repository+'.git')) {
            throw 'The existing origin points to another repository. Review it before publishing.'
        }
        if (-not $existing) { & git remote add origin ($repositoryUrl+'.git') }
        & git push -u origin main
        if ($LASTEXITCODE -ne 0) { throw 'Repository push failed' }
    }
    & gh api --method POST ('repos/'+$Repository+'/pages') -f build_type=workflow 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & gh api ('repos/'+$Repository+'/pages') --jq .build_type
        if ($LASTEXITCODE -ne 0) { throw 'Enable GitHub Pages using GitHub Actions in repository Settings.' }
    }
    if (-not $SkipData) {
        $assetPath = (Resolve-Path -LiteralPath $AssetDirectory).Path
        $manifest = Get-Content -LiteralPath 'metadata\release-assets.json' -Raw | ConvertFrom-Json
        $assets = @()
        foreach ($item in $manifest) {
            $file = Join-Path $assetPath $item.name
            if (-not (Test-Path -LiteralPath $file)) { throw ('Release file missing: '+$item.name) }
            $assets += $file
        }
        $assets += Join-Path $assetPath 'SHA256SUMS-active.txt'
        & gh release view v0.1 --repo $Repository 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            & gh release create v0.1 --repo $Repository --title 'Benchmark source and feature release v0.1' --notes-file RELEASE_NOTES.md @assets
        } else {
            & gh release upload v0.1 --repo $Repository @assets
        }
        if ($LASTEXITCODE -ne 0) { throw 'Release asset upload failed; inspect the release before retrying.' }
    }
    Write-Output ('Repository: '+$repositoryUrl)
    Write-Output ('Pages workflow: '+$repositoryUrl+'/actions/workflows/pages.yml')
} finally { Pop-Location }
