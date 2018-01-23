function net = net_structure_text (codelens)

run matconvnet-beta23/matlab/vl_setupnn.m

net = dagnn.DagNN();

    net.addLayer('conv1', dagnn.Conv('size', [3 1 1 80],...
   	    'hasBias', true, 'stride', [1,1], 'pad', [0 0 0 0]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('collapsing',dagnn.Collapsing(),{'conv1'},{'collaps'},{});
	net.addLayer('pool', dagnn.K_max_pooling(),{'collaps'}, {'pool1'}, {});
	net.addLayer('fc1', dagnn.Conv('size', [2 1 80 codelens],... 
	    'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool1'}, {'fc1'},  {'fc1f'  'fc1b'});
		
		
		
	% init layers	
    net.params(net.getParamIndex('conv1f')).value = 0.01*randn(3,1,1,80,'single');
    net.params(net.getParamIndex('conv1b')).value = 0.01*randn(1,80,'single');
	
	net.params(net.getParamIndex('fc1f')).value = 0.01*randn(2,1,80,codelens,'single');
    net.params(net.getParamIndex('fc1b')).value = 0.01*randn(1,codelens,'single');