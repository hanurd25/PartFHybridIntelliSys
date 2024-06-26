%%Sources

% This mathworks page has been used for inspiration for the neuro-fuzzy system:
% https://se.mathworks.com/help/fuzzy/anfis.html

%The foundation used for the rule base 2 system can be found here:
%https://github.com/hanurd25/partDAssignmentFuzzyLogic/blob/main/ruleBase2.m

%%THIS IS THE CODE FOR A RULE BASE 2 SYSTEM
%% Develop a FIS using command line in matlab
%mamfis is for Mamdani while sugfis is for sugeno
fis = mamfis('Name', "HVACMamdaniRuleBase2");
%% Create a Mamdani fuzzy inference system .
%%fis = mamfis('Name',"TipMamdani");

fis = addInput(fis,[0 0.7],"Name","humidity");
fis = addInput(fis,[0 1.00],"Name","temperature");
fis = addInput(fis,[0 1.00],"Name","degree_Of_utility");

%Adding the membership functions for humidity

% quality levels using Gaussian membership functions.
%gaussmf is for normal distribution
%trimf is for triangular membership function

fis = addMF(fis,"humidity","trapmf",[-0.410 0 0.15 0.3],"Name","veryLow");
fis = addMF(fis,"humidity","trimf",[0.1 0.3 0.5],"Name","low");
fis = addMF(fis,"humidity","trapmf",[0.4 0.55 0.8 0.9],"Name","moderate");
%% Plot the membership functions for the first output variable.
plotmf(fis,"input",1);
%% Add membership functions temperature
%The temperature functions are good
% levels using trapezoidal membership functions.
%T
fis = addMF(fis,"temperature","trapmf",[-0.6 0.0 0.4 0.6],"Name","low");
fis = addMF(fis,"temperature","trimf",[0.4 0.6 0.8],"Name","moderate");
fis = addMF(fis,"temperature","trapmf",[0.6 0.8 1 1.4],"Name","high");
%% Plot the membership functions for the second output variable.
plotmf(fis,"input",2);

%% Adding the membership functions for degree_Of_utility
% levels using trapezoidal membership functions.
fis = addMF(fis,"degree_Of_utility","trapmf",[-0.5 0.0 0.17 0.35],"Name","low");
fis = addMF(fis,"degree_Of_utility","trimf",[0.3 0.5 0.7],"Name","moderate");
fis = addMF(fis,"degree_Of_utility","trapmf",[0.6 0.8 1 1.4],"Name","high");



%% Add the output variable for the HVAC PLANT OUTPUT.
fis = addOutput(fis,[0 1],"Name","HVAC_PLANT_OUTPUT");
%% Add membership functions for the output.

fis = addMF(fis,"HVAC_PLANT_OUTPUT","trapmf",[-0.5 0.0 0.2 0.4],"Name","veryLow");
fis = addMF(fis,"HVAC_PLANT_OUTPUT","trimf",[0.0 0.2 0.4],"Name","low");
fis = addMF(fis,"HVAC_PLANT_OUTPUT","trimf",[0.25 0.35 0.45],"Name","ratherLow");
fis = addMF(fis,"HVAC_PLANT_OUTPUT","trimf",[0.3 0.5 0.7],"Name","moderate");
fis = addMF(fis,"HVAC_PLANT_OUTPUT","trimf",[0.55 0.65 75],"Name","ratherHigh");
fis = addMF(fis,"HVAC_PLANT_OUTPUT","trimf",[0.6 0.8 1],"Name","high");
fis = addMF(fis,"HVAC_PLANT_OUTPUT","trapmf",[0.7 0.9 1 1.4],"Name","veryHigh");
%%
%%plotmf(fis,"output",1);

rule1 = "if humidity is veryLow and degree_Of_utility is low and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule2 = "if humidity is low and degree_Of_utility is low and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule3 = "if humidity is moderate and degree_Of_utility is low and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule4 = "if humidity is veryLow and degree_Of_utility is moderate and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule5 = "if humidity is low and degree_Of_utility is moderate and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule6 = "if humidity is moderate and degree_Of_utility is moderate and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule7 = "if humidity is veryLow and degree_Of_utility is high and temperature is low then HVAC_PLANT_OUTPUT is low" 
rule8 = "if humidity is low and degree_Of_utility is high and temperature is low then HVAC_PLANT_OUTPUT is low"
rule9 = "if humidity is moderate and degree_Of_utility is high and temperature is low then HVAC_PLANT_OUTPUT is veryLow"
rule10 = "if humidity is veryLow and degree_Of_utility is low and temperature is moderate then HVAC_PLANT_OUTPUT is low"
rule11 = "if humidity is low and degree_Of_utility is low and temperature is moderate then HVAC_PLANT_OUTPUT is veryLow"
rule12 = "if humidity is moderate and degree_Of_utility is low and temperature is moderate then HVAC_PLANT_OUTPUT is veryLow"
rule13 = "if humidity is veryLow and degree_Of_utility is moderate and temperature is moderate then HVAC_PLANT_OUTPUT is ratherLow"
rule14 = "if humidity is low and degree_Of_utility is moderate and temperature is moderate then HVAC_PLANT_OUTPUT is low"
rule15 = "if humidity is moderate and degree_Of_utility is moderate and temperature is moderate then HVAC_PLANT_OUTPUT is veryLow"
rule16 = "if humidity is veryLow and degree_Of_utility is high and temperature is moderate then HVAC_PLANT_OUTPUT is moderate"
rule17 = "if humidity is low and degree_Of_utility is high and temperature is moderate then HVAC_PLANT_OUTPUT is ratherLow"
rule18 = "if humidity is moderate and degree_Of_utility is high and temperature is moderate then HVAC_PLANT_OUTPUT is low" 
rule19 = "if humidity is veryLow and degree_Of_utility is low and temperature is high then HVAC_PLANT_OUTPUT is veryHigh"
rule20 = "if humidity is low and degree_Of_utility is low and temperature is high then HVAC_PLANT_OUTPUT is high"
rule21 = "if humidity is moderate and degree_Of_utility is low and temperature is high then HVAC_PLANT_OUTPUT is moderate"
rule22 = "if humidity is veryLow and degree_Of_utility is moderate and temperature is high then HVAC_PLANT_OUTPUT is moderate"
rule23 = "if humidity is low and degree_Of_utility is moderate and temperature is high then HVAC_PLANT_OUTPUT is moderate"
rule24 = "if humidity is moderate and degree_Of_utility is moderate and temperature is high then HVAC_PLANT_OUTPUT is low"
rule25 = "if humidity is veryLow and degree_Of_utility is high and temperature is high then HVAC_PLANT_OUTPUT is ratherHigh"
rule26 = "if humidity is low and degree_Of_utility is high and temperature is high then HVAC_PLANT_OUTPUT is moderate"
rule27 = "if humidity is moderate and degree_Of_utility is high and temperature is high then HVAC_PLANT_OUTPUT is ratherLow"

rules = [rule1 rule2 rule3 rule4 rule5 rule6 rule7 rule8 rule9 rule10 rule11 rule12 rule13 rule14 rule15 rule16 rule17 rule18 rule19 rule20 rule21 rule22 rule23 rule24 rule25 rule26 rule27]


   
%% Add the rules to the FIS.
for i = 1:length(rules)
    fis = addRule(fis, rules{i}); 
end
%% Display the rules
fis.Rules
%% Plotting the membership functions
plotmf(fis, 'input', 1);
plotmf(fis, 'input', 2);
plotmf(fis, 'input', 3);

%% plotfis displays the whole system as a block diagram.
%plotfis(fis);
%ruleview(fis)
%% Generate fuzzy inference system output surface
%surfview(fis);% or gensurf(fis)
%% Random Input data
numSamples = 1000;
humidity = rand(numSamples, 1) * 0.7;  
temperature = rand(numSamples, 1) * 1.00;  
degree_of_utility = rand(numSamples, 1) * 1.00;  
inputs = [humidity, temperature, degree_of_utility];

%% Adjusting the inputs
inputs = min(max(inputs, [0 0 0]), [0.7 1 1]); 

%% Process adjusted inputs through FIS
outputs = zeros(numSamples, 1);
for i = 1:numSamples
    outputs(i) = evalfis(fis, inputs(i, :));
end


%%
cv = cvpartition(numSamples, 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);


%% THis is where i applying indices
inputsTrain = inputs(idxTrain, :);
outputsTrain = outputs(idxTrain);
inputsTest = inputs(idxTest, :);


%These are the targets
outputsTest = outputs(idxTest);


% Create initial FIS structure from training data
%opt = anfisOptions('InitialFIS', 3, 'EpochNumber', 100, 'ValidationData', [inputsTest, outputsTest]);

%EpochNumber      : This sets the maximum number of training epochs/iterations
%InitialStepSize  : This sets the initial step size for the training process to


opt = anfisOptions('InitialFIS', 3, 'EpochNumber', 200, 'InitialStepSize', 0.01, 'ValidationData', [inputsTest, outputsTest], 'DisplayANFISInformation', 1)


% Training the model
[anfisModel, trainError, stepSize, chkFIS] = anfis([inputsTrain, outputsTrain], opt);
%[anfisModel, trainError, stepSize, chkFIS] = anfis([inputsTrain, outputsTrain], options);


predictedOutputs = evalfis(anfisModel, inputsTest);


mse = mean((outputsTest - predictedOutputs).^2);
rmse = sqrt(mse);
meanError = mean(outputsTest - predictedOutputs);
stdDeviation = std(outputsTest - predictedOutputs);

fprintf('MSE: %f\n', mse);
fprintf('RMSE: %f\n', rmse);
fprintf('Mean Error: %f\n', meanError);
fprintf('Standard Deviation: %f\n', stdDeviation);


figure;
plot(outputsTest, 'bo-', 'DisplayName', 'Actual Outputs');
hold on;
plot(predictedOutputs, 'ro-', 'DisplayName', 'Predicted Outputs');
title('Comparison of Actual and Predicted Outputs');
xlabel('Sample Index');
ylabel('HVAC Output');
legend show;
grid on;


outOfRangeInput = [0.8, 1.2, 1.2]; % Beyond specified input ranges
%fisOutput = evalfis(fis, outOfRangeInput);
%nfOutput = evalfis(anfisModel, outOfRangeInput);
%fprintf('FIS Output: %.4f\nNeuro-Fuzzy Output: %.4f\n', fisOutput, nfOutput);

%% Adjust 
adjustedInput = min(max(outOfRangeInput, [0 0 0]), [0.7 1 1]);

%% Evaluating the FIS with sets of adjusted inputs

outOfRangeInputs = [
    0.8, 1.2, 1.2;  
    0.9, 1.3, 1.1;  
    1.0, 1.5, 1.5;  
    1.2, 2.0, 2.0;  
    -0.1, -0.2, -0.1  
];


fisOutput = evalfis(fis, adjustedInput);
nfOutput = evalfis(anfisModel, adjustedInput);


fisOutputs = zeros(size(outOfRangeInputs, 1), 1);
nfOutputs = zeros(size(outOfRangeInputs, 1), 1);

%Evaluating the FIS and Neuro with adjusted inputs


for i = 1:size(outOfRangeInputs, 1)
    adjustedInput = min(max(outOfRangeInputs(i, :), [0 0 0]), [0.7 1 1]);

  
    fisOutputs(i) = evalfis(fis, adjustedInput);
    nfOutputs(i) = evalfis(anfisModel, adjustedInput);
end





figure;
bar([fisOutputs, nfOutputs]);
title('Comparison of FIS and neuro Outputs for sets with adjusted inputs');
xlabel('Input set');
ylabel('Output Value');
legend('FIS', 'Neuro-fuzzy');
set(gca, 'XTick', 1:size(outOfRangeInputs, 1), 'XTickLabel', {'Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5'});

%% Evaluating without adjusted inputs

fisOutputs = zeros(size(outOfRangeInputs, 1), 1);
neuroFisOutputs = zeros(size(outOfRangeInputs, 1), 1);

%Evaluating for each input
for i = 1:size(outOfRangeInputs, 1)
    fisOutputs(i) = evalfis(fis, outOfRangeInputs(i, :));
    neuroFisOutputs(i) = evalfis(anfisModel, outOfRangeInputs(i, :));
end

figure;
bar([fisOutputs, neuroFisOutputs]);
title('FIS and NEURO-FIS Outputs for inputs which is out of range');
xlabel('Input Set');
ylabel('Output Value');
legend('FIS', 'Neuro-FIS');
set(gca, 'XTick', 1:size(outOfRangeInputs, 1), 'XTickLabel', {'Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5'});
grid on;


figure;
plot(outputsTest, predictedOutputs, 'ko'); % Plot predicted vs. actual as points
hold on;
%Ccreating a reference line
plot([min(outputsTest) max(outputsTest)], [min(outputsTest) max(outputsTest)], 'b--');
title('Predicted v actual outputs');
xlabel('Actual');
ylabel('Predicted');
grid on;
legend('Predicted v Actual', 'y = x', 'Location', 'best');

%% Plotting training errors to monitor performance over epochs
figure;
plot(trainError);
title('Training error over epochs/iterations');
xlabel('Epoch/iterations');
ylabel('Training error');
grid on;





figure;
bar([fisOutputs, nfOutputs]);
title('Comparison of FIS and neuro Outputs for sets with adjusted inputs');
xlabel('Input set');
ylabel('Output');
legend('FIS', 'Neuro-fuzzy');
set(gca, 'XTick', 1:size(outOfRangeInputs, 1), 'XTickLabel', {'Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5'});



%% Display how the neural fuzzy system would look
plotfis(anfisModel);
%% Display comparison plot or any further analysis
figure;
plot(outputsTest, 'bo-', 'DisplayName', 'Actual output');
hold on;
plot(predictedOutputs, 'ro-', 'DisplayName', 'Predicted');
title('Comparison of actual and predicted outputs');
xlabel('Sample');
ylabel('HVAC output');
legend show;
grid on;

%% Plotting training errors to display performance over epochs
figure;
plot(trainError);
title('Training error over epochs/iterations');
xlabel('Epoch/iterations');
ylabel('Training error');
grid on;
