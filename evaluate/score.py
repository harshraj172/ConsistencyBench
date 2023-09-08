import numpy as np
from scipy.stats import entropy

from agreements import Agreement

def semantic_clustering(inp, outs, agreement_fn, threshold=0.5,):
    
    C = [[outs[0]]]
    outs = outs[1:]
    for i in range(len(outs)):
        STORED = False
        for j in range(len(C)):
            s_c = C[j][0]
            left_score = agreement_fn(inp, s_c, outs[i])
            right_score = agreement_fn(inp, outs[i], s_c)
            
            if left_score>threshold and right_score>threshold:
                STORED = True
                C[j].append(outs[i])
        if not STORED: C.append([outs[i]])
    return C


class ConsistencyScorer:
    def __init__(self, agreements_list, scoring_type=""):
        super(ConsistencyScorer, self).__init__()
        self.agreements_list = agreements_list
        self.scoring_type = scoring_type
        
    def entropy_score(self, input, outputs, agreement_fn, threshold, _):
        # TODO
        # Add exact score via entropy estimate through Monte Carlo
        clusters = semantic_clustering(input, outputs, agreement_fn, threshold)

        pk = np.array([len(c) for c in clusters])/sum([len(c) for c in clusters])
        H = entropy(pk, base=2)
        return H
    
    def pairwise_score(self, outputs, agreement_fn, threshold, binary=True):
        agreements = 0
        for i, output_i in enumerate(outputs):
            for j, output_j in enumerate(outputs):
                if i == j:
                    continue
                agreement_score = agreement_fn(output_i, output_j)
                if binary and agreement_score >= threshold:
                    agreements += 1
                elif binary == False:
                    agreements += agreement_score
        if (len(outputs) * (len(outputs) - 1)) == 0:
            return 0
        return (1 / (len(outputs) * (len(outputs) - 1))) * agreements
    
    def score(self, input, outputs):
        if self.scoring_type=="entropy":
            scorer = self.entropy_score 
        elif self.scoring_type=="pairwise":
            scorer = self.pairwise_score 
        else:
            raise Exception(f"scoring type '{self.scoring_type}' not available")
        
        con_scores = {}
        for name, threshold in self.agreements_list:
            fn = Agreement(name)
            print('Getting score for ', name)
            con_scores[name] = scorer(input, outputs, fn, threshold, False)
        return con_scores
