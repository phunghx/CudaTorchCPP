local C = ccn2.C

local test2, parent = torch.class('ccn2.test2', 'nn.Module')

function test2:__init(kW, dW)
  parent.__init(self)

  self.kW = kW
  self.dW = dW or kW

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()

end

function test2:updateOutput(input)
  return C['my_test_func2']()
end


function test2:updateGradInput(input, gradOutput)  
 return  input
end
