% show_time_loss

% load('logs/loss.mat');

nb = length(10000:2000:28000);
loss_data = cell(nb, 1);
% k = 1;
% for i=10000:2000:28000
%     val_name = strcat('lossV', num2str(i));
%     tmp = eval(val_name);
%     loss_data{k} = [tmp(:, 1) - tmp(1, 1), tmp(:, 2)];
%     k = k + 1;
% end
% 
% colmap = hsv(nb);
% figure; hold on;
% for i = 1:nb
%     plot(loss_data{i}(3:20:end, 1), loss_data{i}(3:20:end, 2),...
%         'Color', colmap(i,:));
% end
k = 1;
ppl_data = cell(nb, 1);
for i=10000:2000:28000
    val_name = strcat('pplV', num2str(i));
    tmp = eval(val_name);
    ppl_data{k} = [tmp(:, 1) - tmp(1, 1), tmp(:, 2)];
    k = k + 1;
end

hold off;
for i = 1:nb
    figure;
    t = linspace(0, 12, size(ppl_data{i},1)); 
    plotyy(t, ppl_data{i}(:, 1), t, ppl_data{i}(:, 2));
end