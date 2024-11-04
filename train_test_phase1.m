clear all; clc; close all;
load('new_combined_matrix.mat');
load('Project_data.mat')
load('new_combined_matrix_Test.mat')

%%
% RBF
num_folds = 5;
spread_range = 0.1:0.1:2.0; 
hidden_neurons_range = 35:45; 

best_accuracy = 0;
best_spread = 0;
best_hidden_neurons = 0;
best_net = [];

for spread = spread_range
    for hidden_neurons = hidden_neurons_range
        ACC = 0;
        for p = 1:num_folds
            train_indices = [1:(p-1)*110+1,p*110+1:550] ;
            valid_indices = (p-1)*110+1:p*110 ;

            train_data_fold = new_combined_matrix(:, train_indices);
            train_labels_fold = TrainLabels(train_indices);
            
            valid_data_fold = new_combined_matrix(:, valid_indices);
            valid_labels_fold = TrainLabels(valid_indices);

            net = newrb(train_data_fold, train_labels_fold, 0, spread, hidden_neurons);

            validation_predict = sim(net, valid_data_fold);
            validation_predict = sign(validation_predict);
            diff = validation_predict - valid_labels_fold;

            ACC = ACC + 110 - sum(abs(diff)) / 2 ;
        end

        average_accuracy = ACC / 5.5; 

        if average_accuracy > best_accuracy
            best_accuracy = average_accuracy;
            best_spread = spread;
            best_hidden_neurons = hidden_neurons;
        end
    end
end

% Display the best hyperparameters
best_accuracy
best_spread
best_hidden_neurons

%%
% Train the final model on the entire training dataset using the best hyperparameters
spread = best_spread; 
hidden_neurons = best_hidden_neurons;

netrbf = newrb(new_combined_matrix, TrainLabels, 0, spread, hidden_neurons);
train_predict = sim(netrbf, new_combined_matrix);
train_predict = sign(train_predict);
diff = train_predict - TrainLabels;
ACC = 100 - sum(abs(diff)) / 11;

ACC

%%
% Hyperparameter Tuning
num_folds = 5;
best_N = -1;
best_accuracy = 0;

for N = 34:50
    total_accuracy = 0;

    for p = 1:num_folds
        train_indices = [1:(p-1)*110,p*110+1:550] ;
        valid_indices = (p-1)*110+1:p*110 ;

        train_data = new_combined_matrix(:, train_indices);
        train_labels = TrainLabels(1, train_indices);
        valid_data = new_combined_matrix(:, valid_indices);
        valid_labels = TrainLabels(valid_indices);
        
        % Train and validate MLP
        accuracy = trainTestMLP(N, train_data, train_labels, valid_data, valid_labels);
        
        total_accuracy = total_accuracy + accuracy;
    end

    ACCMLP(N) = total_accuracy / num_folds;

    if ACCMLP(N) > best_accuracy
        best_accuracy = ACCMLP(N);
        best_N = N;
    end
end

fprintf('Best N: %d\n', best_N);
fprintf('Best Accuracy: %.2f%%\n', best_accuracy);

%%
% Train Final MLP Model
final_N = best_N;
max_epochs = 1000; 
epochs = 0;
target_accuracy = 70; 

while epochs < max_epochs

    netmlp = fitnet(final_N, 'trainscg'); 

    netmlp = train(netmlp, new_combined_matrix, TrainLabels);

    train_predict = netmlp(new_combined_matrix);
    train_predict = sign(train_predict);
    diff = train_predict - TrainLabels;
    ACC = 100 - sum(abs(diff)) / 11;
    
    if ACC >= target_accuracy
        break; 
    end
    
    epochs = epochs + 1;
end

fprintf('Final Training Accuracy: %.2f%%\n', ACC);


%%
% Test MLP
test_predict_mlp = netmlp(new_combined_matrix_Test);
test_predict_label_mlp = sign(test_predict_mlp);

% Test RBF
test_predict_rbf = sim(netrbf, new_combined_matrix_Test);
test_predict_label_rbf = sign(test_predict_rbf);

% Concatenate MLP and RBF test predictions vertically
test_predict_labels = [test_predict_label_mlp; test_predict_label_rbf];

% Save MLP and RBF test predictions
save('test_predict_labels.mat', 'test_predict_labels');

sum(abs(mean(test_predict_labels)))



%%
function accuracy = trainTestMLP(N, train_data, train_labels, valid_data, valid_labels)
    netmlp = fitnet(N, 'trainscg'); % You can adjust training algorithms
    netmlp = train(netmlp, train_data, train_labels);
    
    validation_predict = netmlp(valid_data);
    validation_predict = sign(validation_predict);
    diff = validation_predict - valid_labels;
    
    accuracy = 100 - sum(abs(diff)) / 2.2;
end




