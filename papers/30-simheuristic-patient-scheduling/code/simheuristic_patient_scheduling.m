%% ========================================================================
%  Paper: M. Dehghanimohammadabadi, M. Rezaeiahari, and T. Keyser,
%         "Simheuristic of Patient Scheduling Using Table-Experiment
%         Approach," in Proc. Winter Simulation Conference (WSC), 2017.
%
%  Description: This code implements two metaheuristic approaches for
%               patient scheduling optimization integrated with Simio:
%               (1) Simulated Annealing (SA) - continuous neighborhood
%               (2) Tabu Search (TS) - permutation-based neighborhood
%               Both methods optimize patient sequencing to minimize
%               average wait time via simulation evaluation.
%
%  Requirements: MATLAB
%                Simio Simulation Software
%                RunExperimentDLLMD.exe (Simio-MATLAB interface)
%                Patients.xlsx (patient data interface)
%                FinalModel.spfx (Simio patient scheduling model)
%
%  Usage: Run simheuristic_sa() for Simulated Annealing approach.
%         Run simheuristic_ts() for Tabu Search approach.
% =========================================================================


%% ########################################################################
%  APPROACH 1: SIMULATED ANNEALING (SA)
%  ########################################################################

function simheuristic_sa()

clc;
clear;
close all;

global NFE;
NFE = 0;

%% --- Problem Definition ---
CostFunction = @(q) MyCost(q);
nVar = 21;                      % Number of Decision Variables
VarSize = [1 nVar];

%% --- SA Parameters ---
MaxIt = 100;                    % Maximum Number of Outer Iterations
MaxIt2 = 100;                   % Maximum Number of Inner Iterations
T0 = 10;                       % Initial Temperature
alpha = 0.98;                   % Temperature Damping Rate

%% --- Initialization ---
x.Position = CreateRandomSolution(nVar);
[x.Cost, x.Sol] = CostFunction(x.Position);
BestSol = x;
BestCost = zeros(MaxIt, 1);
nfe = zeros(MaxIt, 1);
T = T0;

%% --- SA Main Loop ---
for it = 1:MaxIt
    for it2 = 1:MaxIt2

        % Create Neighbor Solution
        xnew.Position = CreateNeighbor(x.Position);
        [xnew.Cost, xnew.Sol] = CostFunction(xnew.Position);

        if xnew.Cost <= x.Cost
            x = xnew;                  % Accept better solution
        else
            % Accept worse solution with probability p
            delta = xnew.Cost - x.Cost;
            p = exp(-delta / T);
            if rand <= p
                x = xnew;
            end
        end

        % Update Best Solution
        if x.Cost <= BestSol.Cost
            BestSol = x;
        end
    end

    BestCost(it) = BestSol.Cost;
    nfe(it) = NFE;
    disp(['Iteration ' num2str(it) ': NFE = ' num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);

    % Cool Down Temperature
    T = alpha * T;
end

%% --- Plot Results ---
figure;
plot(nfe, BestCost, 'LineWidth', 2);
xlabel('NFE');
ylabel('Best Cost');
title('SA Convergence');

end


%% ########################################################################
%  APPROACH 2: TABU SEARCH (TS)
%  ########################################################################

function simheuristic_ts()

clc;
clear;
close all;

global NFE;
NFE = 0;

%% --- Problem Definition ---
CostFunction = @(p) MyCost(p);
nVar = 21;                              % Number of Decision Variables
ActionList = CreatePermActionList(nVar); % All possible moves
nAction = numel(ActionList);

%% --- TS Parameters ---
MaxIt = 10;                             % Maximum Number of Iterations
TL = round(0.05 * nAction);            % Tabu Length

%% --- Initialization ---
sol.Position = randperm(nVar);
sol.Cost = CostFunction(sol.Position);
BestSol = sol;
BestCost = zeros(MaxIt, 1);
TC = zeros(nAction, 1);                % Tabu Counters

%% --- TS Main Loop ---
for it = 1:MaxIt
    bestnewsol.Cost = inf;

    % Evaluate All Non-Tabu Moves
    for i = 1:nAction
        if TC(i) == 0
            newsol.Position = DoAction(sol.Position, ActionList{i});
            newsol.Cost = CostFunction(newsol.Position);
            newsol.ActionIndex = i;
            if newsol.Cost <= bestnewsol.Cost
                bestnewsol = newsol;
            end
        end
    end

    % Update Current Solution
    sol = bestnewsol;

    % Update Tabu List
    for i = 1:nAction
        if i == bestnewsol.ActionIndex
            TC(i) = TL;
        else
            TC(i) = max(TC(i) - 1, 0);
        end
    end

    % Update Best Solution
    if sol.Cost <= BestSol.Cost
        BestSol = sol;
    end

    BestCost(it) = BestSol.Cost;
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
end

%% --- Plot Results ---
BestCost = BestCost(1:it);
figure;
plot(BestCost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
title('Tabu Search Convergence');

end


%% ========================================================================
%  Helper: Cost Function
%  Updates Excel with patient sequence, runs Simio, returns cost.
% =========================================================================
function [z, sol] = MyCost(q)
    global NFE;
    NFE = NFE + 1;
    UpdateExcelInput(q);
    z = RunSimio();
    sol.q = q;
    sol.z = z;
end


%% ========================================================================
%  Helper: Update Excel Input
%  Converts permutation to patient types and writes to Excel.
% =========================================================================
function UpdateExcelInput(q)
    fileName = 'Patients.xlsx';
    worksheet = 'Type';
    r1 = 2;    % Ratio of Follow-up Patients
    r2 = 1;    % Ratio of New Patients
    nVar = size(q, 2);
    Delimiter = r1/(r1+r2) * nVar;
    PatientType = zeros(nVar, 1);
    PatientType(q <= Delimiter) = 1;
    xlswrite(fileName, PatientType, worksheet, 'A2');
end


%% ========================================================================
%  Helper: Run Simio
%  Executes patient scheduling simulation and returns objective value.
% =========================================================================
function Obj = RunSimio()
    filePath = 'C:\Users\m.dehghani\Documents\2 - Publications\5 - Conferences\1- WSC\2017\1- Breast Cancer Center\Simulation Model';
    fileName = 'test.spfx';
    Scenarios = '001;HH2;1';

    Results = ExecuteSimio(filePath, fileName, Scenarios);
    disp(Results);

    if size(Results{1}, 2) > 4
        cost_string1 = strsplit(Results{1}, ';');
        Results = cost_string1{1, 3};
        Obj = str2num(Results(1:6));
    else
        Obj = 1000;     % Penalty for failed simulation
        disp('No Results');
    end
end


%% ========================================================================
%  Helper: Execute Simio
%  Low-level Simio-MATLAB interface via DLL/EXE bridge.
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


%% ========================================================================
%  Helper: Create Random Solution
%  Generates a random permutation as initial solution.
% =========================================================================
function q = CreateRandomSolution(nVar)
    q = randperm(nVar);
end


%% ========================================================================
%  Helper: Create Neighbor (SA)
%  Applies random swap, reversion, or insertion to current solution.
% =========================================================================
function qnew = CreateNeighbor(q)
    m = randi([1 3]);
    switch m
        case 1
            qnew = Swap(q);
        case 2
            qnew = Reversion(q);
        case 3
            qnew = Insertion(q);
    end
end

function qnew = Swap(q)
    n = numel(q);
    i = randsample(n, 2);
    qnew = q;
    qnew([i(1) i(2)]) = q([i(2) i(1)]);
end

function qnew = Reversion(q)
    n = numel(q);
    i = randsample(n, 2);
    i1 = min(i(1), i(2));
    i2 = max(i(1), i(2));
    qnew = q;
    qnew(i1:i2) = q(i2:-1:i1);
end

function qnew = Insertion(q)
    n = numel(q);
    i = randsample(n, 2);
    i1 = i(1);
    i2 = i(2);
    if i1 < i2
        qnew = [q(1:i1-1) q(i1+1:i2) q(i1) q(i2+1:end)];
    else
        qnew = [q(1:i2) q(i1) q(i2+1:i1-1) q(i1+1:end)];
    end
end


%% ========================================================================
%  Helper: Create Permutation Action List (TS)
%  Generates all possible swap, reversion, and insertion moves.
% =========================================================================
function ActionList = CreatePermActionList(n)
    nSwap = n*(n-1)/2;
    nReversion = n*(n-1)/2;
    nInsertion = n^2;
    nAction = nSwap + nReversion + nInsertion;
    ActionList = cell(nAction, 1);
    c = 0;

    % Swap Moves
    for i = 1:n-1
        for j = i+1:n
            c = c + 1;
            ActionList{c} = [1 i j];
        end
    end

    % Reversion Moves
    for i = 1:n-1
        for j = i+1:n
            if abs(i-j) > 2
                c = c + 1;
                ActionList{c} = [2 i j];
            end
        end
    end

    % Insertion Moves
    for i = 1:n
        for j = 1:n
            if abs(i-j) > 1
                c = c + 1;
                ActionList{c} = [3 i j];
            end
        end
    end
    ActionList = ActionList(1:c);
end


%% ========================================================================
%  Helper: Do Action (TS)
%  Applies a specific move (swap/reversion/insertion) to a permutation.
% =========================================================================
function q = DoAction(p, a)
    switch a(1)
        case 1
            q = DoSwap(p, a(2), a(3));
        case 2
            q = DoReversion(p, a(2), a(3));
        case 3
            q = DoInsertion(p, a(2), a(3));
    end
end

function q = DoSwap(p, i1, i2)
    q = p;
    q([i1 i2]) = p([i2 i1]);
end

function q = DoReversion(p, i1, i2)
    q = p;
    q(i1:i2) = p(i2:-1:i1);
end

function q = DoInsertion(p, i1, i2)
    if i1 < i2
        q = [p(1:i1-1) p(i1+1:i2) p(i1) p(i2+1:end)];
    else
        q = [p(1:i2) p(i1) p(i2+1:i1-1) p(i1+1:end)];
    end
end
