function stat = read_stats(filename)
if nargin == 0
    filename = '../data/wiki-stats.mat';
end
s = load(filename);
stat = s.len_stat;