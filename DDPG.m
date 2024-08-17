clear;
clc;
% 
% % Define global parameters
num_episodes = 1024;
max_steps = ceil(0.3 / 0.00001); % Assuming Tf = 0.3 and Ts = 0.00001

% Define environment parameters
mdl = 'DCDC_BBC_RL';
agentblk = [mdl '/RL Agent'];
numObs = 3; % Number of observations [v0, e, de/dt]

% Observation and action specifications
observationInfo = rlNumericSpec([numObs, 1], 'LowerLimit', [-inf -inf 0]', 'UpperLimit', [0.1 110 inf]');
% observationInfo.Name = 'observations';
observationInfo.Description = 'integrated error, error, and measured height';
actionInfo = rlNumericSpec([1 1], 'LowerLimit', 0, 'UpperLimit', 1);
actionInfo.Name = 'action';

% Create the RL environment
env = rlSimulinkEnv(mdl, agentblk, observationInfo, actionInfo);
env.ResetFcn = @(in) setVariable(in, 'init_action', 1);

% Create random initial weights for the networks
actorFC1Weights = randn(256, numObs);
actorFC2Weights = randn(256, 256);
actorOutputWeights = randn(1, 256);

criticFC1Weights = randn(256, numObs + 1); % Note: numObs + 1 for action input
criticFC2Weights = randn(256, 256);
criticOutputWeights = randn(1, 256);

% Define actor and critic networks for DDPG
actorNetwork = [
    imageInputLayer([numObs 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(256, 'Name', 'ActorFC1', 'BiasLearnRateFactor', 0, 'Weights', actorFC1Weights, 'Bias', zeros(256,1))
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(256, 'Name', 'ActorFC2', 'BiasLearnRateFactor', 0, 'Weights', actorFC2Weights, 'Bias', zeros(256,1))
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(1, 'Name', 'action', 'BiasLearnRateFactor', 0, 'Weights', actorOutputWeights, 'Bias', zeros(1,1))
    tanhLayer('Name', 'ActorTanh')
];

criticNetwork = [
    imageInputLayer([numObs + 1 1 1], 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(256, 'Name', 'CriticFC1', 'BiasLearnRateFactor', 0, 'Weights', criticFC1Weights, 'Bias', zeros(256,1))
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(256, 'Name', 'CriticFC2', 'BiasLearnRateFactor', 0, 'Weights', criticFC2Weights, 'Bias', zeros(256,1))
    reluLayer('Name', 'CriticRelu2')
    fullyConnectedLayer(1, 'Name', 'CriticOutput', 'BiasLearnRateFactor', 0, 'Weights', criticOutputWeights, 'Bias', zeros(1,1))
];

% Create actor and critic representations
actor = rlDeterministicActorRepresentation(actorNetwork, observationInfo, actionInfo);
critic = rlQValueRepresentation(criticNetwork, observationInfo, actionInfo);

% Define DDPG agent options
agentOpts = rlDDPGAgentOptions(...
    'SampleTime', 0.00001, ...
    'TargetSmoothFactor', 1e-3, ...
    'ExperienceBufferLength', 100000, ...
    'MiniBatchSize', 64, ...
    'DiscountFactor', 0.99, ...
    'ActorLearnRate', 1e-4, ...
    'CriticLearnRate', 1e-3, ...
    'TargetUpdateFrequency', 1, ...
    'ResetExperienceBufferBeforeTraining', false ...
);

% Create DDPG agent
agent = rlDDPGAgent(actor, critic, agentOpts);

% Define training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', num_episodes, ...
    'MaxStepsPerEpisode', max_steps, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', inf, ...
    'ScoreAveragingWindowLength', 50 ...
);

% Train the agent
trainingStats = train(agent, env, trainOpts);
