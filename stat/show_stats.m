stat_all = read_stats;
stat_lt600 = stat(stat < 600);
stat_le96  = stat_lt600(stat_lt600 <= 96);
sstat_eq1  = sum(stat_le96 == 1);
sstat_eq2  = sum(stat_le96 == 2);
sstat_eq3  = sum(stat_le96 == 3);
sstat_gt96 = sum(stat_all > 96);
sstat_gt80 = sum(stat_all > 80);
sstat_gt64 = sum(stat_all > 64);
stat_3t96  = stat_lt600(stat_lt600 <= 96 & stat_lt600 >=3);
stat_3t64  = stat_le96(stat_le96 <=64 & stat_le96 >=3);
sstat_3t8  = sum(stat_3t64 <= 8);
sstat_9t12 = sum(stat_3t64 <= 12 & stat_3t64 >=9);

fprintf('percentage of length 1: %.2f%%\n', 100*double(sstat_eq1)/length(stat_all));
fprintf('percentage of length 2: %.2f%%\n', 100*double(sstat_eq2)/length(stat_all));
fprintf('percentage of length 3: %.2f%%\n', 100*double(sstat_eq3)/length(stat_all));
fprintf('percentage of length above 96: %.2f%%\n', 100*double(sstat_gt96)/length(stat_all));
fprintf('percentage of length above 80: %.2f%%\n', 100*double(sstat_gt80)/length(stat_all));
fprintf('percentage of length above 64: %.2f%%\n', 100*double(sstat_gt64)/length(stat_all));
fprintf('percentage of length 3 to 96: %.2f%%\n', 100*double(length(stat_3t96))/length(stat_all));
fprintf('percentage of length 3 to 8: %.2f%%\n', 100*double(sstat_3t8)/length(stat_all));
fprintf('percentage of length above 80: %.2f%%\n', 100*double(sstat_9t12)/length(stat_all));

figure;
hist(stat_all, 1:2500);
title('histogram of sentences'' lengths between 1 to max');

figure;
hist(stat_lt600, 1:600);
title('histogram of sentences'' lengths between 1 to 599');

figure;
hist(stat_le96, 1:96);
title('histogram of sentences'' length between 1 to 96');

figure;
hist(stat_3t96, 3:96);
title('histogram of sentences'' length between 3 to 96');

figure;
hist(stat_3t64, 3:64);
title('histogram of sentences'' length between 3 to 64');
