% Run the recorded WavLM cross-corpus DAOD configuration.

clear; clc; close all;

%% ----------------- Algorithm Parameters -------------------
options.p = 10;
options.sigma = 1;
options.lambda = 50;
options.rho = 1;
options.T = 10;
options.t = 0.50;
options.alpha = 0.4;
options.gamma = 0.2;

%% ----------------- Feature Database Configuration ------------------
feature_sets = {
'wavlm',  '/seu_nvme/home/220242615/wavlm'
};

dataset_list = {'M.mat', 'C.mat', 'E.mat', 'Me.mat'};

known_classes = [0, 1, 2];
unknown_class_label = max(known_classes) + 1;

%% ----------------- Loop Over Feature Databases ------------------
for f = 1:size(feature_sets, 1)

    feature_name = feature_sets{f, 1};
    data_dir = feature_sets{f, 2};

    fprintf('\n\n=============================================\n');
    fprintf('Running feature database: %s\n', feature_name);
    fprintf('Data directory: %s\n', data_dir);
    fprintf('=============================================\n');

    result_dir = fullfile(data_dir, ['DAOD_results_', feature_name]);

    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    results = {};
    result_rows = {};
    exp_id = 1;

    %% ----------------- Experiment Loop ------------------------
    for i = 1:length(dataset_list)
        for j = 1:length(dataset_list)

            if i == j
                continue;
            end

            src_name = dataset_list{i};
            tgt_name = dataset_list{j};

            src_file = fullfile(data_dir, src_name);
            tgt_file = fullfile(data_dir, tgt_name);

            [~, src_short, ~] = fileparts(src_name);
            [~, tgt_short, ~] = fileparts(tgt_name);

            fprintf('\n===== [%s] Exp %d: %s -> %s =====\n', ...
            feature_name, exp_id, src_name, tgt_name);

            %% ------------ Check File Existence -----------------
            if ~exist(src_file, 'file')
                warning('Source file does not exist: %s', src_file);
                continue;
            end

            if ~exist(tgt_file, 'file')
                warning('Target file does not exist: %s', tgt_file);
                continue;
            end

            %% ------------ Load Source Dataset -----------------
            [Xs, Ys, ~] = load_feature_mat(src_file);
            Xs = double(Xs);
            Ys = double(Ys(:));

            %% ------------ Load Target Dataset -----------------
            [Xt, Yt, ~] = load_feature_mat(tgt_file);
            Xt = double(Xt);
            Yt = double(Yt(:));

            %% ------------ Basic Dimension Check ----------------
            if size(Xs, 1) ~= length(Ys)
                error('Source feature-label mismatch: %s, size(Xs,1)=%d, length(Ys)=%d', ...
                src_file, size(Xs, 1), length(Ys));
            end

            if size(Xt, 1) ~= length(Yt)
                error('Target feature-label mismatch: %s, size(Xt,1)=%d, length(Yt)=%d', ...
                tgt_file, size(Xt, 1), length(Yt));
            end

            %% ------------ Sort Samples by Label ---------------
            Zs = [Xs, Ys];
            Zs = sortrows(Zs, size(Zs, 2));
            Xs = Zs(:, 1:end-1);
            Ys = Zs(:, end);

            Zt = [Xt, Yt];
            Zt = sortrows(Zt, size(Zt, 2));
            Xt = Zt(:, 1:end-1);
            Yt = Zt(:, end);

            %% ------------ Define Known/Unknown Classes --------
            idx_known_source = ismember(Ys, known_classes);
            Xs = Xs(idx_known_source, :);
            Ys = Ys(idx_known_source);

            Yt_original = Yt;
            Yt(~ismember(Yt, known_classes)) = unknown_class_label;

            fprintf('Source samples after known-class filtering: %d\n', size(Xs, 1));
            fprintf('Target samples: %d\n', size(Xt, 1));

            disp('Source label distribution:');
            tabulate(Ys);

            disp('Target label distribution after unknown mapping:');
            tabulate(Yt);

            %% ------------ Run DAOD Algorithm ------------------
            [~, ~, Yt_pred] = DAOD(Xs, Ys, Xt, Yt, options);

            Yt_pred = double(Yt_pred(:));

            % Map predicted unknowns
            Yt_pred(~ismember(Yt_pred, known_classes)) = unknown_class_label;

            %% ------------ Compute Evaluation Metrics ----------
            Acc_OS_val = sum(Yt_pred == Yt) / length(Yt);

            known_idx = ismember(Yt, known_classes);

            if sum(known_idx) > 0
                Acc_OS_star_val = sum(Yt_pred(known_idx) == Yt(known_idx)) / sum(known_idx);
            else
                Acc_OS_star_val = 0;
            end

            unknown_idx = Yt == unknown_class_label;

            if sum(unknown_idx) > 0
                Acc_unknown_val = sum(Yt_pred(unknown_idx) == unknown_class_label) / sum(unknown_idx);
            else
                Acc_unknown_val = 0;
            end

            if Acc_OS_star_val + Acc_unknown_val > 0
                HOS_val = 2 * Acc_OS_star_val * Acc_unknown_val / ...
                (Acc_OS_star_val + Acc_unknown_val);
            else
                HOS_val = 0;
            end

            fprintf('Acc_OS       = %.4f\n', Acc_OS_val);
            fprintf('Acc_OS*      = %.4f\n', Acc_OS_star_val);
            fprintf('Acc_unknown  = %.4f\n', Acc_unknown_val);
            fprintf('HOS          = %.4f\n', HOS_val);

            %% ------------ Confusion Matrix --------------------
            eval_labels = [known_classes, unknown_class_label];
            confusion_matrix = confusionmat(Yt, Yt_pred, 'Order', eval_labels);

            %% ------------ Store Results -----------------------
            results{exp_id}.Exp_ID = exp_id;
            results{exp_id}.feature = feature_name;
            results{exp_id}.source = src_short;
            results{exp_id}.target = tgt_short;
            results{exp_id}.Acc_OS = Acc_OS_val;
            results{exp_id}.Acc_OS_star = Acc_OS_star_val;
            results{exp_id}.Acc_unknown = Acc_unknown_val;
            results{exp_id}.HOS = HOS_val;
            results{exp_id}.confusion_matrix = confusion_matrix;
            results{exp_id}.Yt = Yt;
            results{exp_id}.Yt_original = Yt_original;
            results{exp_id}.Yt_pred = Yt_pred;

            result_rows(end+1, :) = { ...
            exp_id, feature_name, src_short, tgt_short, ...
            Acc_OS_val, Acc_OS_star_val, Acc_unknown_val, HOS_val};

            %% ------------ Save Confusion Matrix ----------------
            cm_file = fullfile(result_dir, sprintf('confusion_matrix_%s_%s_to_%s.mat', ...
            feature_name, src_short, tgt_short));

            save(cm_file, 'confusion_matrix', 'eval_labels');

            exp_id = exp_id + 1;

        end
    end

    %% ----------------- Save Results for Current Feature Set -----------------------
    result_table = cell2table(result_rows, ...
    'VariableNames', {'Exp_ID','Feature','Source','Target', ...
    'Acc_OS','Acc_OS_star','Acc_unknown','HOS'});

    mat_save_path = fullfile(result_dir, sprintf('all_results_%s.mat', feature_name));
    csv_save_path = fullfile(result_dir, sprintf('results_%s.csv', feature_name));

    save(mat_save_path, 'results', 'result_table', 'options', ...
    'known_classes', 'unknown_class_label');

    writetable(result_table, csv_save_path);

    fprintf('\n[%s] All results saved to:\n', feature_name);
    fprintf('MAT: %s\n', mat_save_path);
    fprintf('CSV: %s\n', csv_save_path);

end

disp('All feature databases have been processed.');
