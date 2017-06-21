function [ll,ul,av] = sem(mat)

mat(isnan(mat)) = [];

av=mean(mat);
ll=av-(std(mat)/sqrt(length(mat)));
ul=av+(std(mat)/sqrt(length(mat)));

end

