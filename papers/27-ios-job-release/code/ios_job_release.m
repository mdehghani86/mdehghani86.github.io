%% ========================================================================
%  Paper: N.O. Fernandes, M. Dehghanimohammadabadi, and S.C. Silva,
%         "Iterative Optimization-Based Simulation: A Decision Support
%         Tool for Job Release," in Proc. WorldCIST, 2018.
%
%  Description: This code implements the optimization module of an
%               Iterative Optimization-based Simulation (IOS) framework
%               for job release decisions. It reads workstation loads and
%               processing times from a shared Excel file (interface with
%               Simio), computes job priority scores, and solves an Integer
%               Linear Program (ILP) to determine optimal job release.
%
%  Requirements: MATLAB Optimization Toolbox (intlinprog)
%                Simio Simulation Software (for full IOS loop)
%                ExcelReadWrite.xlsx (shared data interface)
%
%  Usage: Run intelligent_simulation() to execute the optimization.
%         Run clear_simulation() to reset the Excel interface.
% =========================================================================

function intelligent_simulation()

    %% --- Read Input Data from Excel (Simio Interface) ---
    sheet = 1;
    xlRange  = 'G1:L1';      % Current workstation loads (1x6)
    xlRange2 = 'M1:R1';      % Workstation capacity limits (1x6)
    xlRange3 = 'A1:F1000';   % Job processing times matrix (jobs x 6 workstations)

    load = xlsread('ExcelReadWrite.xlsx', sheet, xlRange);
    limite = xlsread('ExcelReadWrite.xlsx', sheet, xlRange2);
    time = xlsread('ExcelReadWrite.xlsx', sheet, xlRange3);

    %% --- Compute Job Priority Scores ---
    % For each job, compute a normalized score based on how much capacity
    % each workstation has remaining relative to the job's processing time.
    aux1 = 0;
    s_res2 = [];

    for j = 1:size(time,1)
        for i = 1:6
            s_res(i) = time(j,i) / (limite(i) - load(i));
        end
        s_res2 = [s_res2; s_res];
        aux1(j) = sum(s_res);  % Total priority score for job j
    end

    %% --- Set Up Integer Linear Program (ILP) ---
    % Objective: maximize total priority of released jobs (negative for min)
    % Constraints: total processing time on each workstation <= remaining capacity
    % Decision variables: binary (0/1) for each job release decision

    matriz_b = limite - load;    % Right-hand side: remaining capacity
    matriz_b = matriz_b';

    matriz_A = time';            % Constraint matrix: processing times transposed

    lb = zeros(size(time,1), 1); % Lower bounds: 0 (don't release)
    up = ones(size(time,1), 1);  % Upper bounds: 1 (release)

    intcon = 1:size(time,1);     % All variables are integers (binary)

    %% --- Solve ILP Using intlinprog ---
    [estado, y] = intlinprog(-aux1, intcon, matriz_A, matriz_b, [], [], lb, up);

    %% --- Write Results Back to Excel (for Simio) ---
    sheet = 1;
    xlRange4 = 'S1';
    xlswrite('ExcelReadWrite.xlsx', estado, sheet, xlRange4)

end


%% ========================================================================
%  Utility Function: Clear Excel Interface
%  Resets the shared Excel file before a new simulation run.
% =========================================================================

function clear_simulation()

    sheet = 1;
    xlRange = 'A1:S1000';
    var = [' '];
    xlswrite('ExcelReadWrite.xlsx', var, sheet, xlRange)

end
