clear all; clc; close all;

load('normalized_stat_feature_matrix.mat');
load('normalized_freq_feature_matrix.mat');
load('Project_data.mat')

good_feel_indices = find(TrainLabels == 1);
bad_feel_indices = find(TrainLabels == -1);

%%
initial_temperature = 1.0;
cooling_rate = 0.99; 
max_iterations = 900;
k=0.001;

num_random_indexes = 30;
random_indexes_stat = randperm(3186, num_random_indexes);
random_indexes_freq = randperm(590, num_random_indexes);
selected_stat_rows = normalized_stat_feature_matrix(random_indexes_stat, :);
selected_freq_rows = normalized_freq_feature_matrix(random_indexes_freq, :);
new_combined_matrix = [selected_stat_rows; selected_freq_rows];
j1=calculateObjective(new_combined_matrix,good_feel_indices,bad_feel_indices);

for i=1:max_iterations

    rand_index_stat_smallchange = randperm(num_random_indexes,1);
    rand_index_freq_smallchange = randperm(num_random_indexes,1);
    slected_index_stat = randperm(3186, 1);
    new_selected_stat_index = random_indexes_stat;
    new_selected_stat_index(rand_index_stat_smallchange) = slected_index_stat;
    
    slected_index_freq = randperm(590, 1);
    new_selected_freq_index = random_indexes_freq;
    new_selected_freq_index(rand_index_freq_smallchange) = slected_index_freq;
    
    new_selected_stat_rows = normalized_stat_feature_matrix(new_selected_stat_index, :);
    new_selected_freq_rows = normalized_freq_feature_matrix(new_selected_freq_index, :);
    new_combined_matrix2 = [new_selected_stat_rows; new_selected_freq_rows];
    j2=calculateObjective(new_combined_matrix2,good_feel_indices,bad_feel_indices);

    if j2>j1
        random_indexes_stat = new_selected_stat_index;
        random_indexes_freq = new_selected_freq_index;
        j1=j2;
    else
        diff=j1-j2;
        p=exp(-(diff)/(k*initial_temperature));
        a=rand;
        if a<p
            random_indexes_stat = new_selected_stat_index;
            random_indexes_freq = new_selected_freq_index;
            j1=j2;
        end
    end

    initial_temperature=initial_temperature*cooling_rate;

end

    disp(j1)
    save('new_selected_stat_index.mat', 'new_selected_stat_index');
    save('new_selected_freq_index.mat', 'new_selected_freq_index');
    new_selected_stat_rows = normalized_stat_feature_matrix(new_selected_stat_index, :);
    new_selected_freq_rows = normalized_freq_feature_matrix(new_selected_freq_index, :);
    new_combined_matrix2 = [new_selected_stat_rows; new_selected_freq_rows];
    save('new_combined_matrix2.mat', 'new_combined_matrix2');


function objective_value = calculateObjective(new_combined_matrix, good_feel_indices, bad_feel_indices)
    % Compute Fisher Quality here
    u0 = mean(new_combined_matrix , 2);
    u1 = mean(new_combined_matrix(:, good_feel_indices), 2);
    u2 = mean(new_combined_matrix(:, bad_feel_indices), 2);
    S1 = zeros(60, 60);
    S2 = zeros(60, 60);

    for i = 1:length(good_feel_indices)
        x = new_combined_matrix(:, good_feel_indices(i));
        S1 = S1 + (x - u1) * (x - u1)';
    end

    for i = 1:length(bad_feel_indices)
        x = new_combined_matrix(:, bad_feel_indices(i));
        S2 = S2 + (x - u2) * (x - u2)';
    end

    Sb = (u1 - u0) * (u1 - u0)' + (u2 - u0) * (u2 - u0)';
    SW = S1 ./ length(good_feel_indices) + S2 ./ length(bad_feel_indices);
    objective_value = trace(Sb) / trace(SW);
    
end

