%% ========================================================================
%  Paper: J. Seif, M. Dehghanimohammadabadi, and A. Yu, "Integrated
%         Preventive Maintenance and Flow Shop Scheduling Under
%         Uncertainty," Flexible Services and Manufacturing Journal, 2019.
%         DOI: https://doi.org/10.1007/s10696-019-09357-4
%
%  Description: This code implements a Genetic Algorithm (GA) for
%               integrated preventive maintenance and flow shop scheduling
%               under uncertainty. The GA optimizes job sequences and
%               maintenance activity schedules simultaneously, considering
%               stochastic processing times and maintenance durations.
%               A Monte Carlo simulation evaluates each candidate solution.
%
%  Requirements: MATLAB (Statistics Toolbox)
%                IBM CPLEX (optional, for benchmark comparison)
%                Experiments.xlsx (experiment parameters)
%
%  Usage: Run preventive_maintenance_ga() to start the experiment.
% =========================================================================

function preventive_maintenance_ga()

clc;
clear;

%% --- Experiment Initial Settings ---
filename = 'Experiments.xlsx';
phase = 4;
nTP = 30;                                       % Number of Test Problems
sheet = strcat('phase', num2str(phase));

switch phase
    case 0
        N = [4,6,8]; DDTF = 4; maxML = 50; POP = [25,50,100,200,400];
    case 1
        N = 4; DDTF = 4; maxML = 50; POP = 400;
    case 2
        N = 5:9; DDTF = 4; maxML = 50; POP = 200;
    case 3
        N = 4; DDTF = 3:4; maxML = 50; POP = 200;
    case 4
        N = 4; DDTF = 4; maxML = [40 50 60]; POP = 200;
end

gaParameter.MaxIt = 100;
gaParameter.MaxStall = floor(0.20 * gaParameter.MaxIt);

ROWs = size(POP,1)*size(N,1)*size(maxML,1)*size(DDTF,1)*nTP;
data = zeros(ROWs, 10);
xlswrite(filename, data(:,1:2), sheet, 'A3')
xlswrite(filename, data(:,3:end), sheet, 'H3')

%% --- Run Experiments ---
row = 0;
for n = N
    for k = maxML
        experiment.maxML = k;
        experiment.N = n;
        experiment.phase = phase;
        for q = DDTF
            experiment.DDTF = q;
            for TP = 1:nTP
                experiment.TP = TP;
                model = CreateModel(experiment);
                for popSize = POP
                    disp(['Test Problem ', num2str(TP), ' for N=', num2str(n), ...
                          ', maxML=', num2str(k), ', DDTF=', num2str(q), ...
                          ', PopSize=', num2str(popSize), ' initialized...'])
                    gaParameter.nPop = popSize;
                    row = row + 1;
                    tStart = tic;
                    BestSol = ga(model, gaParameter);
                    tElapsed = toc(tStart);

                    z = BestSol.Sol.TotalCost;
                    z1 = BestSol.Sol.AvgMAcost;
                    z2 = BestSol.Sol.AvgTardinessCost;
                    it = BestSol.it;

                    data(row,:) = [popSize, n, q, k, TP, tElapsed, z, z1, z2, it];
                    xlswrite(filename, data(row,1:5), sheet, ['A' num2str(2+row)])
                    xlswrite(filename, data(row,6:end), sheet, ['J' num2str(2+row)])
                    disp('*** Test problem completed.')
                end
            end
        end
    end
end
save('data');

end


%% ========================================================================
%  Core: Genetic Algorithm (GA)
%  Evolves job sequences and maintenance schedules.
% =========================================================================
function BestSol = ga(model, gaParameter)

    global NFE;
    NFE = 0;

    CostFunction = @(q) MyCost(q, model);
    nVar = model.nVar;
    VarSize = [1 nVar];
    VarMin = 0;
    VarMax = 1;

    %% GA Parameters
    MaxIt = gaParameter.MaxIt;
    nPop = gaParameter.nPop;
    pc = 0.8;
    nc = 2*round(pc*nPop/2);
    pm = 0.8;
    nm = round(pm*nPop);
    gamma = 0.05;
    mu = 0.03;
    beta = 8;
    StallCounter = 0;
    MaxStall = gaParameter.MaxStall;
    BestSol.Cost = 10^6;

    %% Initialize Population
    empty_individual.Position = [];
    empty_individual.Cost = [];
    pop = repmat(empty_individual, nPop, 1);

    for i = 1:nPop
        pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
        [pop(i).Cost, pop(i).Sol] = CostFunction(pop(i).Position);
    end

    Costs = [pop.Cost];
    [Costs, SortOrder] = sort(Costs);
    pop = pop(SortOrder);
    BestCost = zeros(MaxIt, 1);
    WorstCost = pop(end).Cost;
    nfe = zeros(MaxIt, 1);

    %% GA Main Loop
    for it = 1:MaxIt
        P = exp(-beta*Costs/WorstCost);
        P = P/sum(P);

        % Crossover
        popc = repmat(empty_individual, nc/2, 2);
        for k = 1:nc/2
            i1 = RouletteWheelSelection(P);
            i2 = RouletteWheelSelection(P);
            [popc(k,1).Position, popc(k,2).Position] = ...
                Crossover(pop(i1).Position, pop(i2).Position, gamma, VarMin, VarMax);
            [popc(k,1).Cost, popc(k,1).Sol] = CostFunction(popc(k,1).Position);
            [popc(k,2).Cost, popc(k,2).Sol] = CostFunction(popc(k,2).Position);
        end
        popc = popc(:);

        % Mutation
        popm = repmat(empty_individual, nm, 1);
        for k = 1:nm
            i = randi([1 nPop]);
            popm(k).Position = Mutate(pop(i).Position, mu, VarMin, VarMax);
            [popm(k).Cost, popm(k).Sol] = CostFunction(popm(k).Position);
        end

        % Merge and Sort
        pop = [pop; popc; popm];
        Costs = [pop.Cost];
        [Costs, SortOrder] = sort(Costs);
        pop = pop(SortOrder);
        WorstCost = max(WorstCost, pop(end).Cost);

        % Truncate
        pop = pop(1:nPop);
        Costs = Costs(1:nPop);

        % Update Best and Check Stall
        lastBestCost = BestSol.Cost;
        BestSol = pop(1);
        BestSol.it = MaxIt;
        BestSol.Stall = 'False';

        if lastBestCost == BestSol.Cost
            StallCounter = StallCounter + 1;
        else
            StallCounter = 0;
        end

        BestCost(it) = BestSol.Cost;
        nfe(it) = NFE;

        if StallCounter > MaxStall
            BestSol.Stall = 'true';
            BestSol.it = it;
            break;
        end
    end
    save('BestSol');
end


%% ========================================================================
%  Helper: Create Model
%  Generates stochastic flow shop model with jobs and maintenance data.
% =========================================================================
function model = CreateModel(experiment)
    N = experiment.N;
    TP = experiment.TP;
    DDTF = experiment.DDTF;
    maxML = experiment.maxML;
    M = 3;      % Number of Machines
    L = 3;      % Number of Maintenance Activities
    R = 30;     % Number of Simulation Replications
    nVar = N + (N-1)*M;

    % Generate Job Processing Time Distributions (Triangular)
    JobsTRI_Parameters = zeros(N, 3);
    for j = 1:N
        parameters = randsample(20:120, 3);
        JobsTRI_Parameters(j,:) = sort(parameters);
    end

    % Generate Maintenance Activity Duration Distributions
    MAsTRI_Parameters = zeros(L, 3);
    for k = 1:L
        parameters = randsample(8:30, 3);
        MAsTRI_Parameters(k,:) = sort(parameters);
    end

    % Due Date and Cost Parameters
    eCmax = N*(max(JobsTRI_Parameters(:)) + max(JobsTRI_Parameters(:)));
    jobsDD = [max(JobsTRI_Parameters(:)) + max(JobsTRI_Parameters(:)), round(2*eCmax/DDTF)];
    jobsPC = [10 20];
    MAspc = [150 450];
    MAwfc = 20;
    maxLevels = [4*maxML 5*maxML 6*maxML];
    MA_Types = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 1 1 1];
    MADUDR = [1, 0.85, 0.75];
    PTIR = [1 1; 0.66 1.5; 0.33 2; -5.00 2];

    % Generate Job Data
    empty_job.ST = zeros(1,M);
    empty_job.FT = zeros(1,M);
    empty_job.Dd = 0;
    empty_job.LateP = 0;
    empty_job.TDcost = 0;
    empty_job.Du = zeros(R,M);
    empty_job.MADu = zeros(R,M);
    empty_job.MACost = zeros(R,M);
    empty_job.MA = zeros(L,M);
    empty_job.TRI = zeros(1,3);
    Job = repmat(empty_job, 1, N);

    for j = 1:N
        for i = 1:M
            for rep = 1:R
                Job(j).Du(rep,i) = round(randDist('TRIA', JobsTRI_Parameters(j,1), JobsTRI_Parameters(j,2), JobsTRI_Parameters(j,3)));
            end
        end
        Job(j).Dd = round(randDist('UNI', jobsDD(1), jobsDD(2)));
        Job(j).LateP = round(randDist('UNI', jobsPC(1), jobsPC(2)));
        Job(j).TRI = JobsTRI_Parameters(j,:);
    end

    % Generate Maintenance Activity Data
    empty_MA.MaxLevel = 0;
    empty_MA.wf = 0;
    empty_MA.Level = zeros(1,M);
    empty_MA.count = zeros(1,M);
    empty_MA.sp = zeros(1,M);
    empty_MA.TRI = zeros(1,3);
    empty_MA.Du = zeros(R,M);
    MA = repmat(empty_MA, 1, L);

    for k = 1:L
        for i = 1:M
            for rep = 1:R
                MA(k).Du(rep,i) = round(randDist('TRIA', MAsTRI_Parameters(k,1), MAsTRI_Parameters(k,2), MAsTRI_Parameters(k,2)));
            end
            MA(k).sp(1,i) = round(randDist('UNI', MAspc(1), MAspc(2)));
        end
        MA(k).MaxLevel = maxLevels(k);
        MA(k).wf = MAwfc;
        MA(k).Level(1,:) = maxLevels(k);
        MA(k).TRI = MAsTRI_Parameters(k,:);
    end

    % Encapsulate Model
    model.N = N; model.M = M; model.L = L; model.R = R;
    model.Job = Job; model.MA = MA; model.nVar = nVar;
    model.MA_Types = MA_Types; model.MADUDR = MADUDR; model.PTIR = PTIR;
    model.TP = TP; model.phase = experiment.phase;
    model.DDTF = DDTF; model.maxML = experiment.maxML;
    save('model');

    % Generate CPLEX Benchmark File
    CreateCPLEXFile(model);
end


%% ========================================================================
%  Helper: Cost Function
%  Evaluates job schedule via Monte Carlo simulation.
% =========================================================================
function [z, SimSol] = MyCost(q, model)
    global NFE;
    NFE = NFE + 1;
    R = model.R;
    w1 = 1; w2 = 1; w3 = 10^6;    % Weights: MA cost, Tardiness, Infeasibility

    z1 = zeros(1,R);
    z2 = zeros(1,R);
    z3 = zeros(1,R);
    sol = [];
    for rep = 1:R
        sol = ParseSolution(q, model, rep);
        z1(rep) = sol.TotalMAcost;
        z2(rep) = sol.TotalTardinessCost;
        z3(rep) = sol.InfeasibilityCounter;
    end

    z = mean(w1*z1 + w2*z2 + w3*z3);
    SimSol.AvgMAcost = mean(z1);
    SimSol.AvgTardinessCost = mean(z2);
    SimSol.TotalCost = z;
    SimSol.newQ = sol.newQ;
    SimSol.model = sol.model;
end


%% ========================================================================
%  Helper: Parse Solution
%  Simulates the flow shop with maintenance activities for one replication.
% =========================================================================
function sol = ParseSolution(q, model, rep)
    InfeasibilityCounter = 0;
    N = model.N; M = model.M; L = model.L;

    % Convert q to job sequence + maintenance schedule
    [~, newQ] = sort(q(1:N));
    for k = 1:L
        newQ = [newQ 0 q(N+(k-1)*(N-1)+1:N+k*(N-1))];
    end

    nmodel = UpdateModel(newQ, model, rep);
    Job = nmodel.Job; MA = nmodel.MA; PTIR = nmodel.PTIR;
    JobSequence = newQ(1:N);

    % Flow Shop Simulation with Maintenance
    for i = 1:M
        jobcounter = 0;
        for j = JobSequence
            jobcounter = jobcounter + 1;

            % Calculate average health level
            AvgLevel = 0;
            for k = 1:L
                if Job(j).MA(k,i) == 1
                    MA(k).Level(1,i) = MA(k).MaxLevel;
                end
                AvgLevel = AvgLevel + MA(k).Level(1,i)/MA(k).MaxLevel;
            end
            AvgLevel = AvgLevel / L;

            % Processing time rate based on machine health
            rate = PTIR(find(PTIR(:,1) < AvgLevel, 1, 'first')-1, 2);
            JobDu = Job(j).Du(rep,i) * rate;

            % Calculate Start and Finish Times
            if find(JobSequence == j) == 1
                if i == 1
                    Job(j).ST(1,i) = 0;
                else
                    Job(j).ST(1,i) = Job(j).FT(1,i-1);
                end
                Job(j).FT(1,i) = Job(j).ST(1,i) + Job(j).Du(rep,i);
            else
                previous_job = JobSequence(jobcounter-1);
                if i == 1
                    Job(j).ST(1,i) = Job(previous_job).FT(1,i) + Job(j).MADu(rep,i);
                else
                    Job(j).ST(1,i) = max(Job(j).FT(1,i-1), Job(previous_job).FT(1,i) + Job(j).MADu(rep,i));
                end
                Job(j).FT(1,i) = Job(j).ST(1,i) + JobDu;
            end

            % Deteriorate Maintenance Levels
            for k = 1:L
                MA(k).Level(1,i) = MA(k).Level(1,i) - JobDu;
                if MA(k).Level(1,i) <= 0
                    InfeasibilityCounter = InfeasibilityCounter + 1;
                end
            end
        end
    end

    % Calculate Total Costs
    TotalMAcost = 0;
    TotalTardinessCost = 0;
    for j = 1:N
        TotalMAcost = TotalMAcost + sum(Job(j).MACost(rep,:));
        Job(j).TDcost = max(Job(j).FT(1,M) - Job(j).Dd, 0) * Job(j).LateP;
        TotalTardinessCost = TotalTardinessCost + Job(j).TDcost;
    end

    nmodel.Job = Job; nmodel.MA = MA;
    sol.InfeasibilityCounter = InfeasibilityCounter;
    sol.TotalMAcost = TotalMAcost;
    sol.TotalTardinessCost = TotalTardinessCost;
    sol.newQ = newQ;
    sol.model = nmodel;
end


%% ========================================================================
%  Helper: Update Model
%  Updates job maintenance durations and costs based on schedule.
% =========================================================================
function nmodel = UpdateModel(q, model, rep)
    nmodel = model;
    M = nmodel.M; N = nmodel.N; L = nmodel.L;
    Job = nmodel.Job; MA = nmodel.MA;
    MA_Types = nmodel.MA_Types;
    JobsSequence = q(1:N);
    MADUDR = nmodel.MADUDR;

    for i = 1:M
        for j = JobsSequence
            JobOrder = find(JobsSequence == j, 1, 'first');
            typenumber = q(N + (i-1)*N + JobOrder);
            MAtype = min(floor(typenumber*8), 7);
            row = MAtype + 1;

            % Calculate Maintenance Duration and Cost
            du = 0; sp = 0; wf = 0; rate = 1;
            NumberOfCombinesMAs = sum(MA_Types(row,:));
            if NumberOfCombinesMAs > 0
                rate = MADUDR(NumberOfCombinesMAs);
            end

            Job(j).MA(:,i) = MA_Types(row,:);

            for k = 1:L
                if MA_Types(row,k) == 1
                    du = MA(k).Du(rep,i);
                    sp = MA(k).sp(1,i);
                    wf = MA(k).wf;
                    MA(k).Level(1,i) = MA(k).MaxLevel;
                    MA(k).count(1,i) = MA(k).count(1,i) + 1;
                    Job(j).MADu(rep,i) = Job(j).MADu(rep,i) + du*rate;
                    Job(j).MACost(rep,i) = Job(j).MACost(rep,i) + du*rate*wf + sp;
                end
            end
        end
    end
    nmodel.Job = Job;
    nmodel.MA = MA;
end


%% ========================================================================
%  Helper: Create CPLEX File
%  Generates data file for CPLEX benchmark comparison.
% =========================================================================
function CreateCPLEXFile(model)
    TP = model.TP; N = model.N; M = model.M; L = model.L; R = model.R;
    Job = model.Job; MA = model.MA;
    DDTF = model.DDTF; maxML = model.maxML; phase = model.phase;

    O = 2^L - 1;
    fid = fopen(['17D_CPLEX_N' num2str(N) '-DDTF' num2str(DDTF) ...
                 '-maxML' num2str(maxML) '-TP' num2str(TP) ...
                 '-Phase' num2str(phase) '.dat'], 'w');
    fprintf(fid, '/* Preventive Maintenance Flow Shop - CPLEX Data */\r\n');
    fprintf(fid, 'm=%d;\r\nn=%d;\r\nl=%d;\r\nS=%d;\r\no=%d;\r\nK=%d;\r\n', ...
            M, N, L, R, O, 100000);

    % Probability
    fprintf(fid, 'probability=[');
    fprintf(fid, repmat('%g,', 1, R-1), repmat(1/R, 1, R-1));
    fprintf(fid, '%g];\r\n', 1/R);

    % Due Dates and Penalties
    fprintf(fid, 'd=[');
    for j = 1:N
        if j < N, fprintf(fid, '%d,', Job(j).Dd);
        else, fprintf(fid, '%d];\r\n', Job(j).Dd); end
    end
    fprintf(fid, 'pi=[');
    for j = 1:N
        if j < N, fprintf(fid, '%d,', Job(j).LateP);
        else, fprintf(fid, '%d];\r\n', Job(j).LateP); end
    end

    fprintf(fid, 'MLmax=[%d,%d,%d];\r\n', MA(1).MaxLevel, MA(2).MaxLevel, MA(3).MaxLevel);
    fprintf(fid, 'WF=20;\r\n');
    fclose(fid);
end


%% ========================================================================
%  Helper: Random Distribution Generator
%  Generates random variates from Normal, Triangular, or Uniform dists.
% =========================================================================
function rnd = randDist(Dist, p1, p2, p3)
    switch Dist
        case 'NORM'
            pd = makedist('Normal', p1, p2);
            rnd = random(pd);
        case 'TRIA'
            pd = makedist('Triangular', p1, p2, p3);
            rnd = random(pd);
        case 'UNI'
            rnd = p1 + rand*abs((p2-p1));
    end
end


%% ========================================================================
%  Helper: Crossover
%  Arithmetic crossover with exploration factor gamma.
% =========================================================================
function [y1, y2] = Crossover(x1, x2, gamma, VarMin, VarMax)
    alpha = unifrnd(-gamma, 1+gamma, size(x1));
    y1 = alpha.*x1 + (1-alpha).*x2;
    y2 = alpha.*x2 + (1-alpha).*x1;
    y1 = max(y1, VarMin); y1 = min(y1, VarMax);
    y2 = max(y2, VarMin); y2 = min(y2, VarMax);
end


%% ========================================================================
%  Helper: Mutate
%  Gaussian mutation on a random subset of variables.
% =========================================================================
function y = Mutate(x, mu, VarMin, VarMax)
    nVar = numel(x);
    nmu = ceil(mu*nVar);
    j = randsample(nVar, nmu);
    sigma = 0.1*(VarMax - VarMin);
    y = x;
    y(j) = x(j) + sigma*randn(size(j))';
    y = max(y, VarMin); y = min(y, VarMax);
end


%% ========================================================================
%  Helper: Roulette Wheel Selection
% =========================================================================
function i = RouletteWheelSelection(P)
    r = rand;
    c = cumsum(P);
    i = find(r <= c, 1, 'first');
end


%% ========================================================================
%  Helper: Tournament Selection
% =========================================================================
function i = TournamentSelection(pop, m)
    nPop = numel(pop);
    S = randsample(nPop, m);
    spop = pop(S);
    [~, j] = min([spop.Cost]);
    i = S(j);
end
