clear
clc

stim_type='places2';

load(['study3_',stim_type]);

beauty=zscore(mean(res.beauty,2));
complexity=zscore(mean(res.complexity,2));
order=zscore(mean(res.order,2));

[y1,y2,y12]=VPA_2(complexity,order,beauty);
