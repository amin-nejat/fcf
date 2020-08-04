# %%

resp_measure_name="wilcoxW"
vs,ps=causalityVsResponse(
          responseOutputs[analysisInd][0][resp_measure_name],
          causalPowers[analysisInd],
          responseOutputs[analysisInd][1]['lapses'],
          figuresFolder+analysisIdStrings[analysisInd]+"_respMeasure="+resp_measure_name+".jpg",
          return_output=1
          )