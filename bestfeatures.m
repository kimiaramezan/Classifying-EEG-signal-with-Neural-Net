clear all; clc; close all;

load('normalized_stat_feature_matrix.mat');
load('normalized_freq_feature_matrix.mat');
load('Project_data.mat')

% Extract indices for 'good' and 'bad' labels
good_feeling = find(TrainLabels == 1);
bad_feeling = find(TrainLabels == -1);

%%
numof_good_features = 60;
best = 0;

for p = 1:200000

    num_random_indexes = 30;
    random_indexes_stat = randperm(3186, num_random_indexes);
    random_indexes_freq = randperm(590, num_random_indexes);
    selected_stat_rows = normalized_stat_feature_matrix(random_indexes_stat, :);
    selected_freq_rows = normalized_freq_feature_matrix(random_indexes_freq, :);
    new_combined_matrix = [selected_stat_rows; selected_freq_rows];
        
    u0 = mean(new_combined_matrix , 2);
    u1 = mean(new_combined_matrix(:, good_feeling), 2);
    u2 = mean(new_combined_matrix(:, bad_feeling), 2);
    S1 = zeros(numof_good_features, numof_good_features);
    S2 = zeros(numof_good_features, numof_good_features);

    for i = 1:length(good_feeling)
        x = new_combined_matrix(:, good_feeling(i));
        S1 = S1 + (x - u1) * (x - u1)';
    end

    for i = 1:length(bad_feeling)
        x = new_combined_matrix(:, bad_feeling(i));
        S2 = S2 + (x - u2) * (x - u2)';
    end

    Sb = (u1 - u0) * (u1 - u0)' + (u2 - u0) * (u2 - u0)';
    SW = S1 ./ length(good_feeling) + S2 ./ length(bad_feeling);
    j = trace(Sb) / trace(SW);

    if p < 20000
        if j > best && j ~= Inf
            best = j;
        end
    else
        if j > best && j ~= Inf
            save('random_indexes_stat.mat', 'random_indexes_stat');
            save('random_indexes_freq.mat', 'random_indexes_freq');
            break
        end
    end
end
disp(j)
save('new_combined_matrix.mat', 'new_combined_matrix');





