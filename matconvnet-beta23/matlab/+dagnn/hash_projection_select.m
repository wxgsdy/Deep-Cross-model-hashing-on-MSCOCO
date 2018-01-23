classdef hash_projection_select < dagnn.ElementWise
  methods
    function outputs = forward(self, inputs, params)
        switch(numel(inputs))
            case 1
                outputs{1}=inputs{1};
            case 3
                outputs{1}=inputs{1}.*inputs{2}+(1-inputs{1}).*inputs{3};
            case 7
                nnodes=numel(inputs);
                tree_level=log2(nnodes+1);
                outputs{1}=cast(zeros(size(inputs{1})),'like',inputs{1});
                for index=(nnodes+1)/2:1:nnodes
                    weight_node=index;
                    weight=cast(ones(size(inputs{1})),'like',inputs{1});
                    for level=tree_level:-1:1
                        if(level<tree_level && mod(weight_node,2)==1)
                            weight=weight.*(1-inputs{weight_node});
                        else
                            weight=weight.*inputs{weight_node};
                        end
                        weight_node=floor(weight_node/2);
                        if(weight_node==0)
                            break;
                        end
                    end
                    outputs{1}=outputs{1}+weight;
                end
        end
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        switch(numel(inputs))
            case 1
                derInputs{1}=derOutputs{1};
            case 3
                derInputs{1} = derOutputs{1}.*(inputs{2}-inputs{3});
                derInputs{2}=derOutputs{1}.*inputs{1};
                derInputs{3}=derOutputs{1}.*(1-inputs{1});
            case 7
                nnodes=numel(inputs);
                tree_level=log2(nnodes+1);
                derInputs=cell(size(inputs));
                % internal node
                for index=1:1:(nnodes+1)/2-1
                    derInputs{index}=derOutputs{1};
                    node_level=ceil(log2(index+1));
                    % upon this node,parents
                    weight_parents=cast(ones(size(inputs{1})),'like',inputs{1});
                    traverse_upon_node=index;
                    if(node_level>1)
                        for level=node_level-1:-1:1                            
                            if(mod(traverse_upon_node,2)==1)
                                traverse_upon_node=floor(traverse_upon_node/2);
                                weight_parents=weight_parents.*(1-inputs{traverse_upon_node});
                            else
                                traverse_upon_node=floor(traverse_upon_node/2);
                                weight_parents=weight_parents.*inputs{traverse_upon_node};
                            end
                            if(traverse_upon_node==1)
                                break;
                            end
                        end
                    end
                    % downward this node,children
                    weight_children=cast(zeros(size(inputs{1})),'like',inputs{1});
                    first_child_leaf=index*2^(tree_level-node_level);
                    for child_leaf_n=1:2^(tree_level-node_level)
                        child_leaf_index=first_child_leaf+child_leaf_n-1;
                        weight_child_node=child_leaf_index;
                        weight_children_tmp=cast(ones(size(inputs{1})),'like',inputs{1});
                        for level=tree_level:-1:node_level
                            if(level==tree_level)
                                weight_children_tmp=weight_children_tmp.*inputs{weight_child_node};
                            elseif(level==node_level)
                                if(mod(weight_child_node,2)==1)
                                    weight_children_tmp=weight_children_tmp*(-1);
                                end
                            else
                                if(mod(weight_child_node,2)==1)
                                    weight_children_tmp=weight_children_tmp.*(1-inputs{weight_child_node});
                                else
                                    weight_children_tmp=weight_children_tmp.*inputs{weight_child_node};
                                end
                            end                            
                            weight_child_node=floor(weight_child_node/2);
                        end
                        weight_children=weight_children+weight_children_tmp;
                    end                    
                    derInputs{index}=derInputs{index}.*weight_parents.*weight_children;
                end
                % leaf node
                for index=(nnodes+1)/2:1:nnodes
                    derInputs{index}=derOutputs{1};
                    weight_node=floor(index/2);
                    weight=cast(ones(size(inputs{1})),'like',inputs{1});
                    for level=tree_level-1:-1:1
                        if(level==0)
                            break;
                        end
                        if(mod(weight_node,2)==1)
                            weight=weight.*(1-inputs{weight_node});
                        else
                            weight=weight.*inputs{weight_node};
                        end
                        weight_node=floor(weight_node/2);
                        if(weight_node==0)
                            break;
                        end
                    end
                    derInputs{index}=derOutputs{1}.*weight;
                end
        end
      
      derParams = {} ;
    end

    function obj = hash_projection(varargin)
      obj.load(varargin) ;
    end
  end
end