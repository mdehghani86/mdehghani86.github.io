%% ========================================================================
%  Paper: N. Kabadayi and M. Dehghanimohammadabadi, "Multi-Objective
%         Supplier Selection: Simulation-Optimization with MCDM,"
%         Annals of Operations Research, 2022.
%         DOI: https://doi.org/10.1007/s10479-021-04424-2
%
%  Description: This code implements an NSGA-II algorithm integrated with
%               TOPSIS (multi-criteria decision making) and Simio
%               simulation for multi-objective supplier selection.
%               NSGA-II evolves candidate solutions representing criteria
%               weights and supplier selections. TOPSIS ranks suppliers,
%               and Simio evaluates supply chain performance.
%
%  Requirements: MATLAB (Statistics Toolbox for normc)
%                Simio Simulation Software
%                RunExperimentDLLMD.exe (Simio-MATLAB interface)
%                SupplierInfo.xlsx (supplier data)
%                SCMmodel.spfx (Simio supply chain model)
%
%  Usage: Run supplier_nsga2() to start the optimization.
% =========================================================================

function supplier_nsga2()

clc;
clear;
close all;

%% --- Problem Definition ---
global NFE;
NFE = 0;

model = ReadModel();
n = model.n;                            % Number of Criteria
m = model.m;                            % Number of Suppliers
CostFunction = @(q) MyCost(model, q);

nVar = 2*n;                             % Decision Variables (criteria selection + weights)
VarSize = [1 nVar];
VarMin = 0;
VarMax = 1;
nObj = 3;                               % Number of Objectives

%% --- NSGA-II Parameters ---
MaxIt = 15;             % Maximum Number of Iterations
nPop = 20;              % Population Size

pCrossover = 0.7;
nCrossover = 2*round(pCrossover*nPop/2);

pMutation = 0.4;
nMutation = round(pMutation*nPop);

mu = 0.02;                              % Mutation Rate
sigma = 0.1*(VarMax - VarMin);          % Mutation Step Size

%% --- Initialize Population ---
empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.Sol = [];
empty_individual.Rank = [];
empty_individual.DominationSet = [];
empty_individual.DominatedCount = [];
empty_individual.CrowdingDistance = [];

pop = repmat(empty_individual, nPop, 1);

for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    [pop(i).Sol, pop(i).Cost] = CostFunction(pop(i).Position);
end

% Non-Dominated Sorting and Crowding Distance
[pop, F] = NonDominatedSorting(pop);
pop = CalcCrowdingDistance(pop, F);
[pop, F] = SortPopulation(pop);

%% --- NSGA-II Main Loop ---
for it = 1:MaxIt

    % Crossover
    popc = repmat(empty_individual, nCrossover/2, 2);
    for k = 1:nCrossover/2
        i1 = randi([1 nPop]);
        i2 = randi([1 nPop]);
        [popc(k,1).Position, popc(k,2).Position] = Crossover(pop(i1).Position, pop(i2).Position);
        [popc(k,1).Sol, popc(k,1).Cost] = CostFunction(popc(k,1).Position);
        [popc(k,2).Sol, popc(k,2).Cost] = CostFunction(popc(k,2).Position);
    end
    popc = popc(:);

    % Mutation
    popm = repmat(empty_individual, nMutation, 1);
    for k = 1:nMutation
        i = randi([1 nPop]);
        popm(k).Position = Mutate(pop(i).Position, mu, sigma, VarMin, VarMax);
        [popm(k).Sol, popm(k).Cost] = CostFunction(popm(k).Position);
    end

    % Merge Parent, Crossover, and Mutation Populations
    pop = [pop; popc; popm];

    % Non-Dominated Sorting
    [pop, F] = NonDominatedSorting(pop);
    pop = CalcCrowdingDistance(pop, F);
    [pop, ~] = SortPopulation(pop);

    % Truncate to Population Size
    pop = pop(1:nPop);

    % Re-sort
    [pop, F] = NonDominatedSorting(pop);
    pop = CalcCrowdingDistance(pop, F);
    [pop, F] = SortPopulation(pop);

    % Display Progress
    F1 = pop(F{1});
    disp(['Iteration ' num2str(it) ': F1 Members = ' num2str(numel(F1))]);

    save('pop', 'pop');
end

save('pop', 'pop');
save('F1', 'F1');
disp('Optimization complete.');

end


%% ========================================================================
%  Helper: Read Model
%  Reads supplier decision matrix from Excel.
% =========================================================================
function model = ReadModel()
    DM = xlsread('SupplierInfo.xlsx', 'SupplierData', 'B2:H11');
    J = -1 * ones(1, size(DM,2));
    J(end) = 1;                         % Last criterion is beneficial
    m = size(DM, 1);                    % Number of Alternatives
    n = size(DM, 2);                    % Number of Criteria
    model.m = m;
    model.n = n;
    model.DM = DM;
    model.J = J;
end


%% ========================================================================
%  Helper: Cost Function
%  Runs TOPSIS to rank suppliers, then simulates supply chain in Simio.
% =========================================================================
function [sol, z] = MyCost(model, q)
    global NFE;
    if isempty(NFE), NFE = 0; end
    NFE = NFE + 1;

    sol = TOPSIS(model, q);
    pause(0.05);
    F = RunSimio();
    z = [F(1).z, F(2).z, F(4).z]';     % [AvgInventory, ServiceLevel, TotalCost]
end


%% ========================================================================
%  Helper: TOPSIS
%  Technique for Order of Preference by Similarity to Ideal Solution.
%  Selects best suppliers and their allocation proportions.
% =========================================================================
function sol = TOPSIS(model, q)
    DM = model.DM;
    J = model.J;
    m = model.m;
    n = model.n;
    numCriteria = 3;
    numSupplier = 2;

    % Step 1: Select Criteria based on decision variable
    a = zeros(1, n);
    [~, So] = sort(q(1:end/2), 'descend');
    a(So <= numCriteria) = 1;
    Selection = find(a == 1);

    DM = DM(:, Selection);
    w = q(end/2+1:end);
    w = w(:, Selection);
    J = J(:, Selection);

    % Step 2: Normalize Decision Matrix
    NDM = normc(DM);

    % Step 3: Weighted Decision Matrix
    V = repmat(w, m, 1) .* NDM;
    V_Beneficial = repmat(J, m, 1) .* V;

    % Step 4: Positive and Negative Ideal Solutions
    PIS = abs(max(V_Beneficial, [], 1));
    NIS = abs(min(V_Beneficial, [], 1));

    % Step 5: Separation Distances
    PIS_distance = repmat(PIS, m, 1) - V;
    NIS_distance = V - repmat(NIS, m, 1);
    S_Plus = zeros(1, m);
    S_Minus = zeros(1, m);
    for i = 1:m
        S_Plus(i) = rssq(PIS_distance(i,:));
        S_Minus(i) = rssq(NIS_distance(i,:));
    end

    % Step 6: Relative Closeness to Ideal Solution
    C = S_Minus ./ (S_Plus + S_Minus);

    % Get Best Suppliers and Proportions
    [Csorted, CsortedPosition] = sort(C, 'descend');
    Suppliers = CsortedPosition(1:numSupplier);
    Value = Csorted(1:numSupplier);
    portion = Value / sum(Value);
    portion(1, 1:numSupplier) = roundn(portion(1, 1:numSupplier), -2);
    portion(1, numSupplier) = 1 - sum(portion(1, 1:numSupplier-1));

    sol.Alternatives = Suppliers;
    sol.Portion = portion;
    sol.Criteria = find(a == 1);

    % Write to Excel for Simio
    d = [Suppliers', portion'];
    xlswrite('SupplierInfo.xlsx', d, 'Simio', 'A2');
end


%% ========================================================================
%  Helper: Run Simio
%  Executes Simio supply chain simulation and parses KPIs.
% =========================================================================
function F = RunSimio()
    filePath = 'C:\Users\m.dehghani\Dropbox (NU College of Eng''g)\1- Research\25- Nihan-Mohammad papers\1- Multi-Objective Suplier Selection\2- Models\3- Integrated SO Model';
    fileName = 'SCMmodel.spfx';
    Scenarios = '001;MD;1';

    Results = ExecuteSimio(filePath, fileName, Scenarios);
    Results_string = strsplit(Results{1}, ';');

    f.label = '';
    f.z = 0;
    F = repmat(f, 1, 4);
    for i = 1:4
        j = 2*i;
        F(i).label = Results_string{1, j};
        F(i).z = round(str2num(Results_string{1, j+1}), 4);
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
%  Helper: Non-Dominated Sorting
%  Assigns Pareto fronts (ranks) to the population.
% =========================================================================
function [pop, F] = NonDominatedSorting(pop)
    nPop = numel(pop);
    for i = 1:nPop
        pop(i).DominationSet = [];
        pop(i).DominatedCount = 0;
    end

    F{1} = [];
    for i = 1:nPop
        for j = i+1:nPop
            if Dominates(pop(i), pop(j))
                pop(i).DominationSet = [pop(i).DominationSet j];
                pop(j).DominatedCount = pop(j).DominatedCount + 1;
            end
            if Dominates(pop(j).Cost, pop(i).Cost)
                pop(j).DominationSet = [pop(j).DominationSet i];
                pop(i).DominatedCount = pop(i).DominatedCount + 1;
            end
        end
        if pop(i).DominatedCount == 0
            F{1} = [F{1} i];
            pop(i).Rank = 1;
        end
    end

    k = 1;
    while true
        Q = [];
        for i = F{k}
            for j = pop(i).DominationSet
                pop(j).DominatedCount = pop(j).DominatedCount - 1;
                if pop(j).DominatedCount == 0
                    Q = [Q j];
                    pop(j).Rank = k + 1;
                end
            end
        end
        if isempty(Q), break; end
        F{k+1} = Q;
        k = k + 1;
    end
end


%% ========================================================================
%  Helper: Calculate Crowding Distance
%  Computes crowding distance for diversity preservation.
% =========================================================================
function pop = CalcCrowdingDistance(pop, F)
    nF = numel(F);
    for k = 1:nF
        Costs = [pop(F{k}).Cost];
        nObj = size(Costs, 1);
        n = numel(F{k});

        d = zeros(n, 1);
        for j = 1:nObj
            [cj, so] = sort(Costs(j,:));
            d(so(1)) = inf;
            d(so(end)) = inf;
            for i = 2:n-1
                d(so(i)) = d(so(i)) + (cj(i+1) - cj(i-1)) / (cj(end) - cj(1) + eps);
            end
        end

        for i = 1:n
            pop(F{k}(i)).CrowdingDistance = d(i);
        end
    end
end


%% ========================================================================
%  Helper: Sort Population
%  Sorts by Rank first, then Crowding Distance (descending).
% =========================================================================
function [pop, F] = SortPopulation(pop)
    [~, CDSO] = sort([pop.CrowdingDistance], 'descend');
    pop = pop(CDSO);
    [~, RSO] = sort([pop.Rank]);
    pop = pop(RSO);

    Ranks = [pop.Rank];
    MaxRank = max(Ranks);
    F = cell(MaxRank, 1);
    for r = 1:MaxRank
        F{r} = find(Ranks == r);
    end
end


%% ========================================================================
%  Helper: Dominates
%  Checks Pareto dominance between two solutions.
% =========================================================================
function b = Dominates(x, y)
    if isstruct(x), x = x.Cost; end
    if isstruct(y), y = y.Cost; end
    b = all(x <= y) && any(x < y);
end


%% ========================================================================
%  Helper: Crossover
%  Arithmetic crossover operator.
% =========================================================================
function [y1, y2] = Crossover(x1, x2)
    alpha = rand(size(x1));
    y1 = alpha.*x1 + (1-alpha).*x2;
    y2 = alpha.*x2 + (1-alpha).*x1;
end


%% ========================================================================
%  Helper: Mutate
%  Gaussian mutation on a subset of variables.
% =========================================================================
function y = Mutate(x, mu, sigma, VarMin, VarMax)
    nVar = numel(x);
    nMu = ceil(mu * nVar);
    j = randsample(nVar, nMu);
    xnew = x + sigma .* randn(size(x));
    y = x;
    y(j) = xnew(j);
    y = max(y, VarMin);
    y = min(y, VarMax);
end
