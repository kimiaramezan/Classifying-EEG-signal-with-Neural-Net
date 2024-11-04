clear all; clc; close all;

load('Project_data.mat');
load('new_selected_stat_index.mat');
load('new_selected_freq_index.mat');
load('normalized_stat_feature_matrix_Test.mat');
load('normalized_freq_feature_matrix_Test.mat');
customTestData = TestData(:, 1001:end, :);

selected_stat_rows_Test2 = normalized_stat_feature_matrix_Test(new_selected_stat_index, :);
selected_freq_rows_Test2 = normalized_freq_feature_matrix_Test(new_selected_freq_index, :);
new_combined_matrix_Test2 = [selected_stat_rows_Test2; selected_freq_rows_Test2];

save('new_combined_matrix_Test2.mat', 'new_combined_matrix_Test2');










