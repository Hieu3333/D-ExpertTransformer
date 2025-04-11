from pycocoevalcap.cider.cider import Cider

gts = {
    0: ["matrix fundus photograph of a 57 year old woman with a von hippel lindau"],
    1: ["matrix fundus photograph of a 57 year old woman with a von hippel lindau"],
    2: ["30 year old female pseudo pohs"],
    3: ["30 year old female pseudo pohs"],
    4: ["30 year old female pseudo pohs"],
    5: ["30 year old female pseudo pohs"],
    6: ["30 year old female pseudo pohs"],
    7: ["30 year old female pseudo pohs"],
    8: ["30 year old female pseudo pohs"],
    9: ["30 year old female pseudo pohs"]
}

res = {
    0: ["fundus fundus of of 18 year year old lindau von hippel lindau von on lindau lindau"],
    1: ["fundus photograph of an an year old with with von hippel lindau von hippel lindau von"],
    2: ["od ocular histoplasmosis syndrome with patient"],
    3: ["presumed ocular histoplasmosis syndrome"],
    4: ["od ocular histo scar with scar in scar"],
    5: ["pseudo angiogram 34 year old patient in macula"],
    6: ["pseudo pohs histoplasmosis syndrome with scar"],
    7: ["od pohs histo a old scar"],
    8: ["od with histo scar with subretinal in scar"],
    9: ["presumed ocular histoplasmosis syndrome"]
}

scorer = Cider()
score, scores = scorer.compute_score(gts, res)
print("CIDEr Score:", score)
