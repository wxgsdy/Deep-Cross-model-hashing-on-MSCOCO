function vl_rmpath()
%VL_SETUPNN Setup the MatConvNet toolbox.
%   VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = vl_rootnn() ;
rmpath(fullfile(root, 'matlab')) ;
rmpath(fullfile(root, 'matlab', 'mex')) ;
rmpath(fullfile(root, 'matlab', 'simplenn')) ;
rmpath(fullfile(root, 'matlab', 'xtest')) ;
rmpath(fullfile(root, 'examples')) ;

if ~exist('gather')
  warning('The MATLAB Parallel Toolbox does not seem to be installed. Activating compatibility functions.') ;
  rmpath(fullfile(root, 'matlab', 'compatibility', 'parallel')) ;
end