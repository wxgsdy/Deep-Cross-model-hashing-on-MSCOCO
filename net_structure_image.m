function net = net_structure_image (net,codelens)

run matconvnet-beta23/matlab/vl_setupnn.m

net = dagnn.DagNN();
%net.accumulateParamDers=true;
%net.conserveMemory=false;

% % use pre-trained VGG-F
net_simplenn=load('data/imagenet-vgg-f.mat');
net=net.fromSimpleNN(net_simplenn);

net.renameVar('x0','data');

%net.renameVar('x20','fc8');

removeLayer(net,'fc8');
removeLayer(net,'prob');

%feature_layer='relu7';
%feature_size=4096;

% add the 8th full connet layer then initialize the parameters 

    net.addLayer('fc8', dagnn.Conv('size', [1 1 4096 codelens],...
	  'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'x19'}, {'x20'},  {'fc8f', 'fc8b'});
	
	
 %init fc8
    net.params(net.getParamIndex('fc8f')).value = 0.01*randn(1,1,4096,codelens,'single');
    net.params(net.getParamIndex('fc8b')).value = 0.01*randn(1,codelens,'single');
    