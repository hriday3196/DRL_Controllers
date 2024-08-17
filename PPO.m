clear;
clc;

% GLOBAL PARAMETERS 
% Parameter values
num_episodes = 1024;
numValidationExperiments = 20;

% Buck Boost Converter Parameters
V_source_value = 48;
L_inductance = 10e-6; 
C_capacitance = 40e-3;
R_load = 100;

% Signal Processing Parameters
prev_time = 0;
init_action = 1; 
stopping_criterion = 1000;
threshold1 = 0.4;
threshold2 = 1;
error_threshold = 0.02;

Ts = 0.0005;
Tf = 0.3;
V_ref = 110;

% RL Parameters
miniBatch_percent = 0.8;
learnRateActor = 0.05;
learnRateCritic = 0.05;
criticLayerSizes = [256 256];
actorLayerSizes = [256 256];
discountFactor = 0.995;

max_steps = ceil(Tf / Ts);
ExperienceHorisonLength = 10;
ClipFactorVal = 0.2;
EntropyLossWeightVal = 0.05;
MiniBatchSizeVal = ceil(ExperienceHorisonLength * miniBatch_percent); 
NumEpochsVal = 5; 
DiscountFactorVal = 0.99;

% RL Agent
mdl = 'DCDC_BBC_RL';
open_system(mdl)
agentblk = [mdl '/RL Agent'];

numObs = 3; % [v0, e, de/dt]
observationInfo = rlNumericSpec([numObs, 1], ...
    'LowerLimit', [-inf -inf 0]', ...
    'UpperLimit', [0.1 V_ref inf]');
observationInfo.Name = 'observations';
observationInfo.Description = 'integrated error, error, and measured height';

a = [1, 1]; 
actionInfo = rlNumericSpec(a);

env = rlSimulinkEnv(mdl, agentblk, observationInfo, actionInfo);
env.ResetFcn = @(in) setVariable(in, 'init_action', 1);
num_inputs = numObs;        



% Observation path
obsPath = [
    featureInputLayer(observationInfo.Dimension(1),Name="obsInLyr")
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(25,Name="obsPathOutLyr")
    ];

% Action path
actPath = [
    featureInputLayer(actionInfo.Dimension(1),Name="actInLyr")
    fullyConnectedLayer(25,Name="actPathOutLyr")
    ];

% Common path
commonPath = [
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(1,Name="QValue")
    ];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,obsPath);
criticNetwork = addLayers(criticNetwork,actPath);
criticNetwork = addLayers(criticNetwork,commonPath);

criticNetwork = connectLayers(criticNetwork, ...
    "obsPathOutLyr","add/in1");
criticNetwork = connectLayers(criticNetwork, ...
    "actPathOutLyr","add/in2");
criticNetwork = dlnetwork(criticNetwork);
critic = rlQValueFunction(criticNetwork, ...
    observationInfo,actionInfo, ...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr");
actorNetwork = [
    featureInputLayer(observationInfo.Dimension(1))
    fullyConnectedLayer(3)
    tanhLayer
    fullyConnectedLayer(actionInfo.Dimension(1))
    ];
actorNetwork = dlnetwork(actorNetwork);
actor = rlContinuousDeterministicActor(actorNetwork,observationInfo,actionInfo);
agent = rlDDPGAgent(actor,critic);
agent.SampleTime = Ts;

agent.AgentOptions.TargetSmoothFactor = 1e-3;
agent.AgentOptions.DiscountFactor = 1.0;
agent.AgentOptions.MiniBatchSize = 64;
agent.AgentOptions.ExperienceBufferLength = 1e6; 

agent.AgentOptions.NoiseOptions.Variance = 0.3;
agent.AgentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-03;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-04;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;

%training parameters    
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', num_episodes, ...
    'MaxStepsPerEpisode', max_steps, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', inf, ...
    'ScoreAveragingWindowLength', 50, ...
    'SaveAgentCriteria', "EpisodeReward", ...
    'SaveAgentValue', 50000);
trainingStats = train(agent, env, trainOpts);