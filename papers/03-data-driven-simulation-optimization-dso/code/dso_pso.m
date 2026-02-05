%% ========================================================================
%  Paper: M. Dehghanimohammadabadi, "Data-Driven Simulation-Optimization
%         (DSO): An Efficient Approach," in Proc. Int. Conf. on
%         Optimization and Learning (OLA), Springer, 2022, pp. 117-132.
%
%  Description: This code implements a Particle Swarm Optimization (PSO)
%               algorithm integrated with Simio simulation for a
%               Data-Driven Simulation-Optimization (DSO) framework.
%               The PSO optimizes job priority sequences, and Simio
%               evaluates each candidate solution via simulation.
%
%  Requirements: MATLAB Optimization Toolbox
%                Simio Simulation Software
%                RunExperimentDLLMD.exe (Simio-MATLAB interface)
%
%  Usage: Run dso_pso() to start the optimization.
% =========================================================================

function dso_pso()

clc;
clear;
close all;
t = datetime('now');

%% --- Problem Definition ---
global NFE;
NFE = 0;

model = CreateModel();
CostFunction = @(x) EvaluateSolution(x, model);

nVar = model.nVar;              % Number of Decision Variables
VarSize = [1 nVar];             % Size of Decision Variables Matrix
VarMin = 0;                     % Lower Bound of Variables
VarMax = 1;                     % Upper Bound of Variables

%% --- PSO Parameters ---
MaxIt = 30;         % Maximum Number of Iterations
nPop = 30;          % Population Size (Swarm Size)

% Constriction Coefficients
phi1 = 2.05;
phi2 = 2.05;
phi = phi1 + phi2;
chi = 2 / (phi - 2 + sqrt(phi^2 - 4*phi));
w = chi;            % Inertia Weight
wdamp = 1;          % Inertia Weight Damping Ratio
c1 = chi * phi1;    % Personal Learning Coefficient
c2 = chi * phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax = 0.1 * (VarMax - VarMin);
VelMin = -VelMax;

%% --- Initialize Swarm ---
empty_particle.Position = [];
empty_particle.Cost = [];
empty_particle.F = [];
empty_particle.Velocity = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];
empty_particle.Best.F = [];

particle = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf;

for i = 1:nPop
    particle(i).Position = randi([VarMin, VarMax], VarSize);
    particle(i).Velocity = zeros(VarSize);
    [particle(i).F, particle(i).Cost] = CostFunction(particle(i).Position);

    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    particle(i).Best.F = particle(i).F;

    % Update Global Best
    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end

BestCost = zeros(MaxIt, 1);
nfe = zeros(MaxIt, 1);

%% --- PSO Main Loop ---
for it = 1:MaxIt
    for i = 1:nPop
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*rand(VarSize).*(particle(i).Best.Position - particle(i).Position) ...
            + c2*rand(VarSize).*(GlobalBest.Position - particle(i).Position);

        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);

        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;

        % Velocity Mirror Effect
        IsOutside = (particle(i).Position < VarMin | particle(i).Position > VarMax);
        particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);

        % Apply Position Limits
        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);

        % Evaluation
        [particle(i).F, particle(i).Cost] = CostFunction(particle(i).Position);

        % Update Personal Best
        if particle(i).Cost < particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            particle(i).Best.F = particle(i).F;

            % Update Global Best
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
    end

    BestCost(it) = GlobalBest.Cost;
    nfe(it) = NFE;
    disp(['Iteration ' num2str(it) ': NFE = ' num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    w = w * wdamp;
end

%% --- Plot Results ---
figure;
semilogy(nfe, BestCost, 'LineWidth', 2);
xlabel('NFE');
ylabel('Best Cost');
title('PSO Convergence');
t2 = datetime('now') - t;
disp(['Total Time: ' char(t2)]);

end


%% ========================================================================
%  Helper Function: Create Model
%  Defines the problem structure (number of jobs and machines).
% =========================================================================
function model = CreateModel()
    I = 50;                 % Total number of jobs
    J = 2;                  % Total number of machines
    nVar = I * J;           % Priority number of each job on each machine
    model.nVar = nVar;
    model.I = I;
    model.J = J;
end


%% ========================================================================
%  Helper Function: Evaluate Solution
%  Converts PSO solution to Simio input, runs simulation, returns cost.
% =========================================================================
function [F, z] = EvaluateSolution(q, model)
    global NFE;
    NFE = NFE + 1;

    fileName = 'Case1';
    ParseSolution(q, model, fileName);
    F = RunSimio(fileName);

    w1 = -1/1000;       % Weight for Time in System (minimize)
    w2 = 1/300000;      % Weight for Tardiness (minimize)
    z = w1*F(1) + w2*F(2);
end


%% ========================================================================
%  Helper Function: Parse Solution
%  Converts continuous PSO variables to job priority sequences and writes
%  them to Excel for Simio to read.
% =========================================================================
function ParseSolution(q, model, fileName)
    I = model.I;
    fileName = strcat(fileName, '.xlsx');

    % Determine priority number for each machine
    [~, q1] = sort(q(1:I));
    [~, q2] = sort(q(I+1:100));

    % Write priority sequences to Excel
    sheet = 'MatlabInput';
    xlswrite(fileName, q1', sheet, 'A2')
    xlswrite(fileName, q2', sheet, 'B2')
end


%% ========================================================================
%  Helper Function: Run Simio
%  Executes the Simio simulation model and parses output responses.
% =========================================================================
function F = RunSimio(fileName)
    % Simio file directory (update path as needed)
    filePath = 'C:\Paper2';
    fileName = strcat(fileName, '.spfx');

    % Define scenarios
    Scenarios = '001;MD;1';

    % Execute Simio via external interface
    Results = ExecuteSimio(filePath, fileName, Scenarios);

    % Parse simulation output
    Results_string = strsplit(Results{1}, ';');
    f1 = str2num(Results_string{1,3}(1:6)) * 60;   % Time in System
    f2 = str2num(Results_string{1,5}(1:6));         % Tardiness
    F = [f1, f2];
end


%% ========================================================================
%  Helper Function: Execute Simio
%  Low-level interface to run Simio experiment via DLL/EXE bridge.
%  Note: Do not modify this section.
% =========================================================================
function Results = ExecuteSimio(filePath, fileName, Scenarios)
    addpath(filePath)
    TXTfilePathName = strcat(filePath, '\SimioLink.txt');
    fid = fopen(TXTfilePathName, 'wt');
    fprintf(fid, '%s\r\n', filePath, fileName, Scenarios);
    fclose(fid);

    EXEfilePathName = strcat(filePath, '\RunExperimentDLLMD.exe');
    system(EXEfilePathName);
    Results = textread(TXTfilePathName, '%s');
end
