%% ========================================================================
%  Paper: M. Dehghanimohammadabadi, M. Rezaeiahari, and J. Seif,
%         "Multi-Objective Patient Appointment Scheduling Framework
%         (MO-PASS)," SIMULATION, 2023.
%         DOI: https://doi.org/10.1177/00375497221132574
%
%  Description: This code implements a Multi-Objective Particle Swarm
%               Optimization (MOPSO) algorithm for patient scheduling.
%               The optimizer determines patient types, physician
%               assignments, and inter-arrival times to minimize Length
%               of Stay (LOS) while maximizing the number of patients
%               served. Simio simulation evaluates each schedule.
%
%  Requirements: MATLAB
%                Simio Simulation Software
%                RunExperimentDLLMD.exe (Simio-MATLAB interface)
%
%  Usage: Run mopass_mopso() to start the multi-objective optimization.
% =========================================================================

function mopass_mopso()

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

%% --- MOPSO Parameters ---
MaxIt = 20;             % Maximum Number of Iterations
nPop = 30;              % Population Size
nRep = 15;              % Repository Size

w = 0.5;                % Inertia Weight
wdamp = 0.99;           % Inertia Weight Damping Rate
c1 = 1;                 % Personal Learning Coefficient
c2 = 2;                 % Global Learning Coefficient

nGrid = 5;              % Number of Grids per Dimension
alpha = 0.1;            % Inflation Rate
beta = 2;               % Leader Selection Pressure
gamma = 2;              % Deletion Selection Pressure
mu = 0.1;               % Mutation Rate

%% --- Initialize Population ---
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];
empty_particle.IsDominated = [];
empty_particle.GridIndex = [];
empty_particle.GridSubIndex = [];

pop = repmat(empty_particle, nPop, 1);

for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop(i).Velocity = zeros(VarSize);
    pop(i).Cost = CostFunction(pop(i).Position);
    pop(i).Best.Position = pop(i).Position;
    pop(i).Best.Cost = pop(i).Cost;
end

%% --- Initialize Repository ---
pop = DetermineDomination(pop);
rep = pop(~[pop.IsDominated]);
Grid = CreateGrid(rep, nGrid, alpha);
for i = 1:numel(rep)
    rep(i) = FindGridIndex(rep(i), Grid);
end

%% --- MOPSO Main Loop ---
for it = 1:MaxIt
    for i = 1:nPop
        % Select Leader from Repository
        leader = SelectLeader(rep, beta);

        % Update Velocity
        pop(i).Velocity = w*pop(i).Velocity ...
            + c1*rand(VarSize).*(pop(i).Best.Position - pop(i).Position) ...
            + c2*rand(VarSize).*(leader.Position - pop(i).Position);

        % Update Position
        pop(i).Position = pop(i).Position + pop(i).Velocity;

        % Evaluate New Position
        pop(i).Cost = CostFunction(pop(i).Position);

        % Apply Mutation
        pm = (1 - (it-1)/(MaxIt-1))^(1/mu);
        NewSol.Position = Mutate(pop(i).Position, pm, VarMin, VarMax);
        NewSol.Cost = CostFunction(NewSol.Position);

        if Dominates(NewSol, pop(i))
            pop(i).Position = NewSol.Position;
            pop(i).Cost = NewSol.Cost;
        elseif ~Dominates(pop(i), NewSol)
            if rand < 0.5
                pop(i).Position = NewSol.Position;
                pop(i).Cost = NewSol.Cost;
            end
        end

        % Update Personal Best
        if Dominates(pop(i), pop(i).Best)
            pop(i).Best.Position = pop(i).Position;
            pop(i).Best.Cost = pop(i).Cost;
        elseif ~Dominates(pop(i).Best, pop(i))
            if rand < 0.5
                pop(i).Best.Position = pop(i).Position;
                pop(i).Best.Cost = pop(i).Cost;
            end
        end
    end

    % Update Domination Status
    pop = DetermineDomination(pop);

    % Add Non-Dominated Particles to Repository
    rep = [rep; pop(~[pop.IsDominated])];
    rep = DetermineDomination(rep);
    rep = rep(~[rep.IsDominated]);

    % Update Grid
    Grid = CreateGrid(rep, nGrid, alpha);
    for i = 1:numel(rep)
        rep(i) = FindGridIndex(rep(i), Grid);
    end

    % Truncate Repository if Full
    if numel(rep) > nRep
        Extra = numel(rep) - nRep;
        for e = 1:Extra
            rep = DeleteOneRepMember(rep, gamma);
        end
    end

    % Plot Pareto Front
    figure(1);
    PlotCosts(pop, rep);

    disp(['Iteration ' num2str(it) ': Repository Members = ' num2str(numel(rep))]);
    w = w * wdamp;
end

%% --- Display Results ---
t2 = datetime('now') - t;
disp(['Total Time: ' char(t2)]);

end


%% ========================================================================
%  Helper: Create Model
%  Defines patient scheduling parameters (types, physicians, arrivals).
% =========================================================================
function model = CreateModel()
    PatientsMix = [0.45, 0.30, 0.25];          % Follow-up, SecondOpinion, Consult
    PatientsMixCum = cumsum(PatientsMix);
    Types = size(PatientsMix, 2);
    Patients_Physician = 30;
    nPhysicians = 3;
    nPatients = nPhysicians * Patients_Physician;
    InterarrivalTimes = 10:10:60;
    nVar = nPatients + Types^2;

    Phyciation_Types = repmat(1:nPhysicians, 1, Patients_Physician);

    model.PatientTypes = Types;
    model.PatientsMix = PatientsMix;
    model.nPatients = nPatients;
    model.InterarrivalTimes = InterarrivalTimes;
    model.nVar = nVar;
    model.Patients_Physician = Patients_Physician;
    model.nPhysicians = nPhysicians;
    model.Phyciation_Types = Phyciation_Types;
    model.PatientsMixCum = PatientsMixCum;
end


%% ========================================================================
%  Helper: Evaluate Solution
%  Runs simulation and returns multi-objective costs [LOS, -nPatients].
% =========================================================================
function z = EvaluateSolution(q, model)
    global NFE;
    NFE = NFE + 1;

    fileName = 'Metaheuristic';
    ParseSolution(q, model, fileName);
    F = RunSimio(fileName);

    % F(2)=LOS, F(3)=Num visited patients, F(4)=OverTime
    if F(4) > 0
        F(2) = 1000;       % Penalize overtime solutions
        F(3) = -1000;
    end
    z = [F(2), -F(3)]';    % Minimize LOS, maximize patients
end


%% ========================================================================
%  Helper: Parse Solution
%  Converts MOPSO variables to patient schedule and writes to Excel.
% =========================================================================
function ParseSolution(q, model, fileName)
    PatientsMixCum = model.PatientsMixCum;
    nPatients = model.nPatients;
    InterarrivalTimes = model.InterarrivalTimes;
    Phyciation_Types = model.Phyciation_Types;
    nPhysicians = model.nPhysicians;
    fileName = strcat(fileName, '.xlsx');

    % Initialize patient structures
    Patient.Type = [];
    Patient.PhysicianType = 0;
    Patient.IA = 0;
    Patients = repmat(Patient, nPatients, 1);

    % Determine Patient Type based on cumulative distribution
    for i = 1:nPatients
        if q(i) < PatientsMixCum(1)
            Patients(i).Type = 1;
        elseif q(i) < PatientsMixCum(2)
            Patients(i).Type = 2;
        else
            Patients(i).Type = 3;
        end
        Patients(i).PhysicianType = Phyciation_Types(i);
    end
    q(q < 0) = 0.01;

    % Determine Inter-Arrival Times
    p = q(nPatients+1:end);
    k = size(InterarrivalTimes, 2);
    InterArrivals = reshape(InterarrivalTimes(min(floor(p*(k)+1), k)), [3, 3]);

    for m = 1:nPhysicians
        for k2 = nPhysicians+m:nPhysicians:nPatients
            i = Patients(k2 - nPhysicians).Type;
            j = Patients(k2).Type;
            Patients(k2).IA = InterArrivals(i, j);
        end
    end

    % Write to Excel
    sheet = 'MatlabInput';
    xlswrite(fileName, [Patients.IA]', sheet, 'C2')
    xlswrite(fileName, [Patients.Type]', sheet, 'D2')
    xlswrite(fileName, [Patients.PhysicianType]', sheet, 'E2')
end


%% ========================================================================
%  Helper: Run Simio
%  Executes Simio simulation and parses performance metrics.
% =========================================================================
function F = RunSimio(fileName)
    filePath = 'C:\Test\1-Metaheuristic';
    fileName = strcat(fileName, '.spfx');
    Scenarios = '001;MD;1';

    Results = ExecuteSimio(filePath, fileName, Scenarios);
    Results_string = strsplit(Results{1}, ';');

    f1 = str2num(Results_string{1,3}(1:6))';    % Utilization
    f2 = str2num(Results_string{1,5}(1:6))*60;  % LOS
    f3 = str2num(Results_string{1,7}(1:end));    % Num Patients
    f4 = str2num(Results_string{1,9}(1:end));    % Overtime
    f5 = str2num(Results_string{1,11}(1:4));     % Additional metric
    F = [f1, f2, f3, f4, f5];
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
%  Helper: Dominates
%  Checks if solution x dominates solution y (Pareto dominance).
% =========================================================================
function b = Dominates(x, y)
    if isstruct(x), x = x.Cost; end
    if isstruct(y), y = y.Cost; end
    b = all(x <= y) && any(x < y);
end


%% ========================================================================
%  Helper: Determine Domination
%  Marks each particle as dominated or non-dominated.
% =========================================================================
function pop = DetermineDomination(pop)
    nPop = numel(pop);
    for i = 1:nPop
        pop(i).IsDominated = false;
    end
    for i = 1:nPop-1
        for j = i+1:nPop
            if Dominates(pop(i), pop(j))
                pop(j).IsDominated = true;
            end
            if Dominates(pop(j), pop(i))
                pop(i).IsDominated = true;
            end
        end
    end
end


%% ========================================================================
%  Helper: Create Grid
%  Creates adaptive grid structure for repository management.
% =========================================================================
function Grid = CreateGrid(pop, nGrid, alpha)
    c = [pop.Cost];
    cmin = min(c, [], 2);
    cmax = max(c, [], 2);
    dc = cmax - cmin;
    cmin = cmin - alpha*dc;
    cmax = cmax + alpha*dc;
    nObj = size(c, 1);

    empty_grid.LB = [];
    empty_grid.UB = [];
    Grid = repmat(empty_grid, nObj, 1);
    for j = 1:nObj
        cj = linspace(cmin(j), cmax(j), nGrid+1);
        Grid(j).LB = [-inf cj];
        Grid(j).UB = [cj +inf];
    end
end


%% ========================================================================
%  Helper: Find Grid Index
%  Assigns grid cell index to a particle based on its cost vector.
% =========================================================================
function particle = FindGridIndex(particle, Grid)
    nObj = numel(particle.Cost);
    nGrid = numel(Grid(1).LB);
    particle.GridSubIndex = zeros(1, nObj);
    for j = 1:nObj
        particle.GridSubIndex(j) = find(particle.Cost(j) < Grid(j).UB, 1, 'first');
    end
    particle.GridIndex = particle.GridSubIndex(1);
    for j = 2:nObj
        particle.GridIndex = particle.GridIndex - 1;
        particle.GridIndex = nGrid * particle.GridIndex;
        particle.GridIndex = particle.GridIndex + particle.GridSubIndex(j);
    end
end


%% ========================================================================
%  Helper: Select Leader
%  Selects a leader from the repository using roulette wheel on grid density.
% =========================================================================
function leader = SelectLeader(rep, beta)
    GI = [rep.GridIndex];
    OC = unique(GI);
    N = zeros(size(OC));
    for k = 1:numel(OC)
        N(k) = numel(find(GI == OC(k)));
    end
    P = exp(-beta*N);
    P = P / sum(P);
    sci = RouletteWheelSelection(P);
    sc = OC(sci);
    SCM = find(GI == sc);
    smi = randi([1 numel(SCM)]);
    leader = rep(SCM(smi));
end


%% ========================================================================
%  Helper: Delete One Repository Member
%  Removes a member from the most crowded grid cell.
% =========================================================================
function rep = DeleteOneRepMember(rep, gamma)
    GI = [rep.GridIndex];
    OC = unique(GI);
    N = zeros(size(OC));
    for k = 1:numel(OC)
        N(k) = numel(find(GI == OC(k)));
    end
    P = exp(gamma*N);
    P = P / sum(P);
    sci = RouletteWheelSelection(P);
    sc = OC(sci);
    SCM = find(GI == sc);
    smi = randi([1 numel(SCM)]);
    rep(SCM(smi)) = [];
end


%% ========================================================================
%  Helper: Mutate
%  Applies uniform mutation to a single decision variable.
% =========================================================================
function xnew = Mutate(x, pm, VarMin, VarMax)
    nVar = numel(x);
    j = randi([1 nVar]);
    dx = pm * (VarMax - VarMin);
    lb = max(x(j) - dx, VarMin);
    ub = min(x(j) + dx, VarMax);
    xnew = x;
    xnew(j) = unifrnd(lb, ub);
end


%% ========================================================================
%  Helper: Roulette Wheel Selection
%  Selects an index based on probability distribution P.
% =========================================================================
function i = RouletteWheelSelection(P)
    r = rand;
    C = cumsum(P);
    i = find(r <= C, 1, 'first');
end


%% ========================================================================
%  Helper: Plot Costs
%  Plots the current Pareto front from the repository.
% =========================================================================
function PlotCosts(~, rep)
    rep_costs = [rep.Cost];
    plot(rep_costs(1,:), -rep_costs(2,:), 'k+', 'markersize', 8);
    xlabel('LOS');
    ylabel('Number of Patients');
    title('Pareto Front');
    hold off;
end
