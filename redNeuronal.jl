##------------------------------------------------------------------------------##
##		Amanda Solano Astorga													##
##		Yasiell Vallejos Gomez													##
##------------------------------------------------------------------------------##

function createMatrixWeights(rows,columns)
	weightMatrix = rand(-3:3, rows, columns)
	return weightMatrix
end

function flattenFunction(S)
	flatten = 1.0 ./ (1.0 .+ exp(-S))
	return flatten
end


function multiplyInputsWeights(input, weight)
	inputsWeight = *(input, weight)
	return inputsWeight
end

function changeOuputArray(output)
	newArray = output
	for index in eachindex(output)
		if output[index] >= 0.5
			newArray[index] = 1
		else
			newArray[index] = 0
		end
	end
	return newArray
end

#Evalua si ambos son iguales para romper el ciclo
function compareOutputArray(outputGenerate, correctOutput)
	newOutputArray = changeOuputArray(outputGenerate)
	for index in eachindex(outputGenerate)
		if newOutputArray[index] != correctOutput[index]
			return false
		end
	end
	return true
end


function getChangeFactorFinaleLayer(outputGenerate, outputExpected)
  	changeFactor = ((outputExpected .- outputGenerate) .* outputGenerate .* (1 .- outputGenerate))
	println("ChangePAge> \n", changeFactor)
	return changeFactor
end

function getChangeFactorHiddenLayer(lastOutput, weightHidden, changeFactorFinal)
  	changeFactorHidden = (lastOutput .*(1 .- lastOutput)) .* (*(changeFactorFinal, weightHidden'))
	println("ChangeHidden> \n", changeFactorHidden)
	return changeFactorHidden
end

function updateInputWeight(eta, weightOutputMatrix ,changeFactorFinalLayer, flattenMatrixResults)
	newWeight = weightOutputMatrix .+ eta .* changeFactorFinalLayer .* flattenMatrixResults'
	return newWeight
end

function updateOutputWeight(eta, weightDataBaseMatrix, changeFactorHiddenLayer, X)
	newWeight = weightDataBaseMatrix .+ eta .*changeFactorHiddenLayer .* X'
	return newWeight
end

######????????????????????????????????????????????????????????????####################

function initial(input, output, eta, oculta, error, max_iter)
	coefficientLearning = 1
	dataBaseMatrix = Int64[0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0]
	correctOutput =[1 0 0 0 0 0 0 0 0 0]

	rowsDataBase = size(dataBaseMatrix, 2)
	weightDataBaseMatrix = createMatrixWeights(64,oculta)

	columnsOutput = size(correctOutput, 2)
	weightOutputMatrix = createMatrixWeights(oculta,10)

    for iter = 0 : max_iter 
    	##------------------------ Forward -----------------------------
    
		weightedSumMatrix = multiplyInputsWeights(dataBaseMatrix, weightDataBaseMatrix)
	    flattenMatrixResults = flattenFunction(weightedSumMatrix + coefficientLearning)

	    weightedSumMatrixHidden = multiplyInputsWeights(flattenMatrixResults, weightOutputMatrix)
	    flattenMatrixHiddenResults = flattenFunction(weightedSumMatrixHidden + coefficientLearning)

	    println("Resultado Iter[",iter,"] \n", flattenMatrixHiddenResults)

	    ##------------------------- Back -------------------------------
	    changeFactorFinalLayer = getChangeFactorFinaleLayer(flattenMatrixHiddenResults, correctOutput)
	    changeFactorHiddenLayer = getChangeFactorHiddenLayer(flattenMatrixResults, weightOutputMatrix, changeFactorFinalLayer)

	    updateWeightOutputMatrix = updateInputWeight(eta, weightOutputMatrix, changeFactorFinalLayer, flattenMatrixResults)
	    updateWeightDataBaseMatrix = updateOutputWeight(eta, weightDataBaseMatrix, changeFactorHiddenLayer, dataBaseMatrix)

	    ##------------------------ Verify if is the finalflattenFunction array --------------------
	    equalValue = compareOutputArray(changeFactorHiddenLayer, correctOutput)

	    if equalValue == true
	    	break
	    end

	    weightOutputMatrix = updateWeightOutputMatrix
	    weightDataBaseMatrix = updateWeightDataBaseMatrix

    end

end




initial("input", "output", 0.5, 10, "error", 2500)