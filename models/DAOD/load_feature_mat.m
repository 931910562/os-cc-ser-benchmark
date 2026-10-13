% Load supplied feature matrices and labels from MAT files.

function [features, labels, labels_str] = load_feature_mat(mat_path)
    %LOAD_FEATURE_MAT Load feature files saved in different MAT/HDF5 formats.
    %
    % Supported:
    %   1. Standard MATLAB MAT-file:
    %        features, labels, labels_str
    %   2. MATLAB v7.3 / HDF5 MAT-file:
    %        /features, /labels, /labels_str
    %   3. Different possible field names:
    %        features / feature / feats / X / data
    %        labels / label / y / Y

    labels_str = [];

    feature_keys = {'features', 'feature', 'feats', 'X', 'data'};
    label_keys   = {'labels', 'label', 'y', 'Y'};
    str_keys     = {'labels_str', 'label_str', 'label_names', 'classes'};

    try
        S = load(mat_path);

        features = get_field_by_names(S, feature_keys);
        labels = get_field_by_names(S, label_keys);
        labels_str = get_field_by_names_optional(S, str_keys);

    catch
        fprintf('[WARN] load() failed. Try HDF5 h5read(): %s\n', mat_path);

        info = h5info(mat_path);

        feature_path = find_h5_dataset_path(info, feature_keys);
        label_path   = find_h5_dataset_path(info, label_keys);
        str_path     = find_h5_dataset_path(info, str_keys);

        if isempty(feature_path)
            error('Cannot find features dataset in HDF5 file: %s', mat_path);
        end

        if isempty(label_path)
            error('Cannot find labels dataset in HDF5 file: %s', mat_path);
        end

        features = h5read(mat_path, feature_path);
        labels = h5read(mat_path, label_path);

        if ~isempty(str_path)
            labels_str = h5read(mat_path, str_path);
        else
            labels_str = [];
        end
    end

    features = double(features);
    labels = double(labels(:));

    % Correct feature direction.
    % Expected:
    %   features: [N, D]
    %   labels:   [N, 1]
    if size(features, 1) ~= length(labels) && size(features, 2) == length(labels)
        features = features';
    end

    if size(features, 1) ~= length(labels)
        error('Feature-label size mismatch in file: %s. features=[%d,%d], labels=%d', ...
        mat_path, size(features, 1), size(features, 2), length(labels));
    end
end

function value = get_field_by_names(S, names)

    value = [];
    fields = fieldnames(S);

    for i = 1:length(names)
        idx = find(strcmpi(fields, names{i}), 1);

        if ~isempty(idx)
            value = S.(fields{idx});
            return;
        end
    end

    error('Required field not found. Available fields: %s', strjoin(fields, ', '));
end

function value = get_field_by_names_optional(S, names)

    value = [];
    fields = fieldnames(S);

    for i = 1:length(names)
        idx = find(strcmpi(fields, names{i}), 1);

        if ~isempty(idx)
            value = S.(fields{idx});
            return;
        end
    end
end

function dataset_path = find_h5_dataset_path(info, names)

    dataset_path = [];

    % Search datasets in current group
    for i = 1:length(info.Datasets)

        dname = info.Datasets(i).Name;

        for k = 1:length(names)
            if strcmpi(dname, names{k})

                if strcmp(info.Name, '/')
                    dataset_path = ['/', dname];
                else
                    dataset_path = [info.Name, '/', dname];
                end

                return;
            end
        end
    end

    % Search subgroups recursively
    for g = 1:length(info.Groups)

        dataset_path = find_h5_dataset_path(info.Groups(g), names);

        if ~isempty(dataset_path)
            return;
        end
    end
end
