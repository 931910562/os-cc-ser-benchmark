% Estimate the balance between marginal and conditional adaptation.

function [mu,adist_m,adist_c] = estimate_mu(Xs,Ys,Xt,Yt)
    C = length(unique(Ys));
    list_adist_c = [];
    epsilon = 1e-3;
    for i = 1 : C
        index_i = Ys == i;
        Xsi = Xs(index_i,:);
        index_j = Yt == i;
        Xtj = Xt(index_j,:);
        adist_i = adist(Xsi,Xtj);
        list_adist_c = [list_adist_c;adist_i];
    end
    adist_c = mean(list_adist_c);

    adist_m = adist(Xs,Xt);
    mu = adist_c / (adist_c + adist_m);
    if mu > 1    % Theoretically mu <= 1, but calculation may be over 1
        mu = 1;
    elseif mu <= epsilon
        mu = 0;
    end
end

function dist = adist(Xs,Xt)
    % Return zero when either domain has no samples.
    if isempty(Xs) || isempty(Xt)
        dist = 0;
        return;
    end

    % Assign domain labels: source=1 and target=2.
    Yss = ones(size(Xs,1),1);
    Ytt = ones(size(Xt,1),1) * 2;
    labels = [Yss; Ytt];

    % Require at least two domain classes.
    if numel(unique(labels)) < 2
        dist = 0;
        return;
    end

    % Train a linear SVM to distinguish the domains.
    model_linear = fitclinear([Xs;Xt], labels, 'Learner','svm');

    % Predict domain labels and compute the error.
    ypred = predict(model_linear, [Xs;Xt]);
    error = mae(labels, ypred);

    % Convert the error to the domain-distance estimate.
    dist = 2 * (1 - 2 * error);
end
