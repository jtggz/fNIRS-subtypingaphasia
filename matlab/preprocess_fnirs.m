function preprocess_fnirs(inputDir, outputDir, kitDir)
% Apply the manuscript's NIRS-KIT processing to continuous fNIRS runs.
% Each input MAT must contain nirsdata with oxyData, dxyData, totalData,
% T, and exception_channel. Channel QC is performed before this function.
% Keep input/output directories outside the public code repository.
arguments
    inputDir (1,1) string
    outputDir (1,1) string
    kitDir (1,1) string
end
assert(isfolder(inputDir), 'Input directory does not exist');
assert(isfolder(kitDir), 'NIRS-KIT directory does not exist');
assert(~startsWith(string(java.io.File(outputDir).getCanonicalPath), ...
    string(java.io.File(fileparts(mfilename('fullpath'))).getCanonicalPath)), ...
    'Choose an output directory outside the code repository');
addpath(fullfile(kitDir, 'Subfunctions'));
addpath(fullfile(kitDir, 'Otherfunctions', 'TDDR-master'));
required = {'NR_detrend','NR_motioncorrection','NR_filter','TDDR'};
for j = 1:numel(required)
    assert(~isempty(which(required{j})), 'Missing NIRS-KIT function: %s', required{j});
end
if ~isfolder(outputDir), mkdir(outputDir); end
files = dir(fullfile(inputDir, '*.mat'));
assert(~isempty(files), 'No MAT files found');
for i = 1:numel(files)
    name = files(i).name;
    destination = fullfile(outputDir, name);
    assert(~isfile(destination), 'Output already exists: %s', destination);
    payload = load(fullfile(inputDir, name));
    assert(isfield(payload, 'nirsdata'), 'Missing nirsdata: %s', name);
    n = payload.nirsdata;
    fields = {'oxyData','dxyData','totalData'};
    good = find(n.exception_channel(:) == 0);
    assert(~isempty(good), 'No valid channels: %s', name);
    small = n;
    for j = 1:numel(fields)
        x = n.(fields{j});
        assert(size(x,2) == numel(n.exception_channel), 'Channel mask mismatch');
        assert(all(isfinite(x(:,good)), 'all'), 'Nonfinite valid channel');
        small.(fields{j}) = x(:,good);
    end
    small.nch = numel(good);
    small.exception_channel = zeros(1,numel(good));
    small = NR_detrend(small,1);
    small = NR_motioncorrection(small,1); % TDDR
    small = NR_filter(small,1,3,3,0.01,0.08); % zero-phase LP then HP
    for j = 1:numel(fields)
        x = small.(fields{j});
        assert(all(isfinite(x), 'all'), 'Nonfinite processed signal');
        n.(fields{j})(:,good) = x;
    end
    nirsdata = n; %#ok<NASGU>
    save(destination, 'nirsdata', '-v7');
end
end
