clear all; clc; close all;

load('Project_data.mat');
load('random_indexes_stat.mat')
load('random_indexes_freq.mat')
customTestData = TestData(:, 1001:end, :);
%%
stat_feature_matrix_Test = zeros(3200,159);
freq_feature_matrix_Test = zeros(1000,159);

% statistical features 
% row 1 to 59 is for variance of each channel 
% row 60 to 767 is for AR coefficients of each channel(12 sampels) 
% row 768 to 1357 is for amp histogram of each channel(10 sampels) 
% row 1358 to 1416 is for form factor of each channel
% row 1417 to 1475 is for skewness of each channel
% row 1476 to 3186 is for covariance of channels


for i = 1:159
    counter = 1476;
    for j = 1:59
        stat_feature_matrix_Test(j, i) = var(customTestData(j, :, i));

        % Calculate AR coefficients for each channel (12 samples)
        ar_coeff = calculate_ar_coefficients(customTestData(j, :, i));
        stat_feature_matrix_Test(60 + (j - 1) * 12 : 71 + (j - 1) * 12, i) = ar_coeff;
        
        % Calculate amp histogram for each channel (10 samples)
        amp_hist = calculate_amp_histogram(customTestData(j, :, i));
        stat_feature_matrix_Test(768 + (j - 1) * 10 : 768 + j * 10 - 1, i) = amp_hist;

        % Calculate and store the form factor for each channel
        channel_data = customTestData(j, :, i);  % Extract the channel data
        form_factor = calculateFormFactor(channel_data);
        stat_feature_matrix_Test(1357 + j, i) = form_factor;

        % Calculate Skewness for each channel
        skewness_value = calculateSkewness(customTestData(j, :, i));
        stat_feature_matrix_Test(1416 + j, i) = skewness_value;

        % Calculate Covariance for each pair of channels
        for k = j+1:59
            channel1_data = customTestData(j, :, i);
            channel2_data = customTestData(k, :, i);
            covariance_value = calculateCovariance(channel1_data, channel2_data);
            stat_feature_matrix_Test(counter, i) = covariance_value;
            counter = counter + 1;
        end

    end
end

% frequancy features 
% row 1 to 59 is for max freq of each channel 
% row 60 to 118 is for mid freq each channel
% row 119 to 177 is for mean freq of each channel
% row 178 to 590 is for band power of each channel

for i = 1:159
    for j = 1:59
        channel_data = customTestData(j, :, i);
        maxFreq = findMaxFrequencyUsingFFT(channel_data, fs);
        freq_feature_matrix_Test(j, i) = maxFreq;

        % Calculate mid freq (row 60 to 118)
        midFreq = medfreq(channel_data, fs);
        freq_feature_matrix_Test(j+59, i) = midFreq;
        
        % Calculate mean freq (row 119 to 177)
        meanFreq = meanfreq(channel_data, fs);
        freq_feature_matrix_Test(j+118, i) = meanFreq;

        % Calculate band power (row 178 to 590)
        bandPowers = calculateBandPower(channel_data, fs);
        freq_feature_matrix_Test(7*j+171:7*j+177, i) = bandPowers;

    end
end


% Statistical features
variance_range = 1:59;
ar_coeff_range = 60:767;
amp_histogram_range = 768:1357;
form_factor_range = 1358:1416;
skewness_range = 1417:1475;
covariance_range = 1476:3186;

% Frequency features
max_freq_range = 1:59;
mid_freq_range = 60:118;
mean_freq_range = 119:177;
band_power_range = 178:590;

% Initialize the normalized matrices
normalized_stat_feature_matrix_Test = zeros(size(stat_feature_matrix_Test));
normalized_freq_feature_matrix_Test = zeros(size(freq_feature_matrix_Test));
normalized_stat_feature_matrix_Test = myCustomScaler(stat_feature_matrix_Test, [variance_range, ar_coeff_range, amp_histogram_range, form_factor_range, skewness_range, covariance_range]);
normalized_freq_feature_matrix_Test = myCustomScaler(freq_feature_matrix_Test, [max_freq_range, mid_freq_range, mean_freq_range, band_power_range]);

% Save the normalized matrices
save('normalized_stat_feature_matrix_Test.mat', 'normalized_stat_feature_matrix_Test');
save('normalized_freq_feature_matrix_Test.mat', 'normalized_freq_feature_matrix_Test');


%%
load('normalized_stat_feature_matrix_Test.mat')
load('normalized_freq_feature_matrix_Test.mat')
selected_stat_rows_Test = normalized_stat_feature_matrix_Test(random_indexes_stat, :);
selected_freq_rows_Test = normalized_freq_feature_matrix_Test(random_indexes_freq, :);
new_combined_matrix_Test = [selected_stat_rows_Test; selected_freq_rows_Test];

save('new_combined_matrix_Test.mat', 'new_combined_matrix_Test');



%%


function scaled = myCustomScaler(features, range)
    min_val = min(features(range, :), [], 1);
    max_val = max(features(range, :), [], 1);
    scaled = (features(range, :) - min_val) ./ (max_val - min_val);
end

function ar_coefficient = calculate_ar_coefficients(channel)
    order = 12;  % AR model order
    ar_coefficient = zeros(order, 1);

    for p = 1:order
        sum_val = 0;
        for n = p+1:length(channel)
            sum_val = sum_val + channel(n) * channel(n - p);
        end
        ar_coefficient(p) = sum_val / length(channel);
    end
end

function amp_histogram = calculate_amp_histogram(channel)
    num_bins = 10;
    amp_histogram = zeros(num_bins, 1);

    min_val = min(channel);
    max_val = max(channel);

    bin_edges = linspace(min_val, max_val, num_bins + 1);

    for k = 1:length(channel)
        amp_val = channel(k);
        bin_idx = find(amp_val >= bin_edges(1:end-1) & amp_val <= bin_edges(2:end));
        
        if ~isempty(bin_idx)
            amp_histogram(bin_idx) = amp_histogram(bin_idx) + 1;
        end
    end
end

function form_factor = calculateFormFactor(channel)
    rms_value = rms(channel);
    
    average_value = mean(channel);
    
    form_factor = rms_value / average_value;
end

function skewness_value = calculateSkewness(channel)
    % Calculate the skewness of the input signal channel
    skewness_value = skewness(channel);
end

function covariance_value = calculateCovariance(channel1, channel2)
    % Check if input channels have the same length
    if length(channel1) ~= length(channel2)
        error('Input channels must have the same length');
    end
    
    % Calculate the covariance
    covariance_value = cov(channel1, channel2);
    covariance_value=covariance_value(1,2);
end

function maxFreq = findMaxFrequencyUsingFFT(input_signal, sampling_rate)
    N = length(input_signal); 
    spectrum = fft(input_signal);
    frequencies = (0:N-1) * (sampling_rate / N); 
    [~, index] = max(abs(spectrum(1:N/2))); 
    maxFreq = frequencies(index);
end

function bandPowers = calculateBandPower(channel, fs)
    frequency_ranges = [0 3; 3 7; 7 12; 12 15; 16 20; 20 30; 30 100];
    num_ranges = size(frequency_ranges, 1);
    
    bandPowers = zeros(num_ranges, 1);
    total_power = bandpower(channel, fs, [0 100]);
    
    for i = 1:num_ranges
        range = frequency_ranges(i, :);
        bandPowers(i) = bandpower(channel, fs, range) / total_power;
    end
end









