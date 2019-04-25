from functools import reduce
from random import choice
from itertools import takewhile, product
import json, codecs, csv
from os.path import isfile
from time import localtime, strftime
from joblib import Parallel, delayed

def union(Ss):
    return reduce(set.union, Ss)


# dictionary utilities

def getRandomKey(a_dict, printKey = False):
    randKey = choice(list(a_dict.keys()))
    if printKey:
        print('Random key: {0}'.format(randKey))
    return randKey

def testRandomKey(a_dict, printKey = True, printVal = True):
    randKey = getRandomKey(a_dict)
    if printKey:
        print('Random key: {0}'.format(randKey))
    if printVal:
        print('value ⟶ {0}'.format(a_dict[randKey]))
    return {'key': randKey, 'val': a_dict[randKey]}


def transpose(d, inner_keys, outer_keys):
    '''
    Let 
        As = {'a0', 'a1'}
        Bs = {'b0', 'b1'}
        d = {'a0': {'b0': 0, 'b1': 1}, 
             'a1': {'b0': 0, 'b1': 1}}
    Then
        transpose(d, Bs, As) = 
        {'b0': {'a1': 0, 'a0': 0}, 'b1': {'a1': 1, 'a0': 1}}
        
    Alternative schematization: Let
        f: A ⟶ B = {'a0':'b0','a1':'b1'}
        g: B ⟶ C = {'b0':0,'b1':1}
        d: A ⟶ B ⟶ C = g ⚬ f
    Then 
        d' = transpose(d, Bs, As) = 
        d': B ⟶ A ⟶ C = flip(g ⚬ f)
    '''
    return {inner_key:{outer_key:d[outer_key][inner_key]
                     for outer_key in outer_keys if outer_key in d and inner_key in d[outer_key]}
            for inner_key in inner_keys}

def project_dict(the_dict, keys_to_keep):
    new_dict = {key:the_dict[key] for key in the_dict.keys() if key in keys_to_keep}
    return new_dict

def edit_dict(the_dict, the_key, the_new_value):
    '''
    Composable (because it returns a value) but stateful(= in-place) dictionary update.
    '''
    the_dict.update({the_key: the_new_value})
    return the_dict

def modify_dict(the_dict, the_key, the_new_value):
    '''
    Composable and (naively-implemented) non-mutating dictionary update.
    '''
    new_dict = {k:the_dict[k] for k in the_dict}
    new_dict.update({the_key: the_new_value})
    return new_dict

def filter_dict(d, cond):
    return {k:d[k] for k in d if cond(k, d[k])}

gtZero = lambda k,v: v > 0.0

def mapValues(f, d):
    '''
    Maps k ⟶ f(d[k)) for each k in d.
    '''
    return {k:f(d[k]) for k in d}


# N-phones, k-factors, and other string-related code

leftEdge = '⋊'
rightEdge = '⋉'
edgeSymbols = {leftEdge, rightEdge}


def tupleToDottedString(pair): 
    return '.'.join(pair)

def dottedStringToTuple(s): 
    return tuple(s.split('.'))

t2ds = tupleToDottedString
ds2t = dottedStringToTuple

def loadHammondAligned(hammond_fn = 'Hammond_newdic_IPA_aligned.csv'):
    hammond_newdic = []
    with open(hammond_fn) as csvfile:
        my_reader = csv.DictReader(csvfile, delimiter='\t')
        for row in my_reader:
            #print(row)
            hammond_newdic.append(row)
    return hammond_newdic

def getAlignedHammondEntryMatches(phonword, removeEdgeSymbols = True, hammond_fn = 'Hammond_newdic_IPA_aligned.csv', hammond_dl = None, wordform_field = 'Transcription'):
    if removeEdgeSymbols:
        phonword = t2ds(ds2t(phonword)[1:-1])
    
    if hammond_dl is None:
        hammond_newdic = loadHammondAligned(hammond_fn)
    else:
        hammond_newdic = hammond_dl

    matchingEntries = [entry for entry in hammond_newdic if entry[wordform_field] == phonword]
    return tuple(matchingEntries)

def getAlignedHammondOrthWordMatches(phonword, removeEdgeSymbols = True, hammond_fn = 'Hammond_newdic_IPA_aligned.csv', hammond_dl = None, wordform_field = 'Transcription'):
    matchingEntries = getAlignedHammondEntryMatches(phonword, removeEdgeSymbols, hammond_fn, hammond_dl, wordform_field)
    matchingOrthWords = {entry['Orthography'] for entry in matchingEntries}
    return matchingOrthWords

def toOrth_h(phonword, removeEdgeSymbols = True, hammond_fn = 'Hammond_newdic_IPA_aligned.csv', hammond_dl = None, wordform_field = 'Transcription'):
    return getAlignedHammondOrthWordMatches(phonword, removeEdgeSymbols, hammond_fn, hammond_dl, wordform_field)

# from itertools import takewhile, product
# from random import choice

def dsToKfactors(k, ds):
    seq = ds2t(ds)
    l = len(seq)
    if k > l:
        return tuple()
    kFactor_start_indices = takewhile(lambda pair: pair[0] <= l-k, enumerate(seq))
    kFactors = tuple(seq[index[0]:index[0]+k] for index in kFactor_start_indices)
    return set(map(t2ds, kFactors))

def dsTo2factors(ds):
    return dsToKfactors(2, ds)
def dsTo3factors(ds):
    return dsToKfactors(3, ds)

def lexiconToKfactors(DSs, k):
    myDsToKfactors = lambda ds: dsToKfactors(k, ds)
    return union(map(set, map(myDsToKfactors, DSs)))

def lexiconTo2factors(DSs):
    return union(map(set, map(dsTo2factors, DSs)))
def lexiconTo3factors(DSs):
    return union(map(set, map(dsTo3factors, DSs)))


def compareKfactors(DSs_A, DSs_B, k):
    A = lexiconToKfactors(DSs_A, k)
    B = lexiconToKfactors(DSs_B, k)
    return {"A == B":A == B, "A - B": A - B, "B - A": B - A}

def sameKfactors(DSs_A, DSs_B, k):
    return compareKfactors(DSs_A, DSs_B, k)["A == B"]

def hasIllicitKfactors(W, illicit_k_factors):
    if type(W) == str:      
        # gather the k-factors into an immutable data structure
        illicit_kfs = tuple(illicit_k_factors)
        # get the set of k-factor lengths (values of k) among the illicit_kfs
        illicit_factor_lengths = set([len(ds2t(kf)) for kf in illicit_kfs])
        # map each k to the set of k-factors of dotted string ds
        kFactorSets = {kf_l:dsToKfactors(kf_l, W) for kf_l in illicit_factor_lengths}
        illegal_kfactors_discovered = tuple(ikf for ikf in illicit_kfs if ikf in kFactorSets[len(ds2t(ikf))])
        if illegal_kfactors_discovered == tuple():
            return False
        return illegal_kfactors_discovered
    else:
        myFunc = lambda w: hasIllicitKfactors(w, illicit_k_factors)
        results = tuple(map(myFunc, W))
        if not any(results):
            return False
        return set(t2ds(each) for each in results if each != False)

def sigmaK(sigma, k):
    return product(sigma, repeat=k)

def dsToKfactorSequence(k, ds):
    seq = ds2t(ds)
    l = len(seq)
    if k > l:
        return tuple()
    kFactor_start_indices = takewhile(lambda pair: pair[0] <= l-k, enumerate(seq))
    kFactors = tuple(seq[index[0]:index[0]+k] for index in kFactor_start_indices)
    return tuple(map(t2ds, kFactors))

def threeFactorSequenceToDS(threeFactors):
    wLE = threeFactors[0][0]
    wRE = threeFactors[-1][-1]
    w_NE = '.'.join([ds2t(eachTriphone)[1] for eachTriphone in threeFactors])
    return '.'.join([wLE, w_NE, wRE])

def randomString(sigma, l, hasLeftEdge=True):
    s_t = tuple([choice(list(sigma)) for each in range(l)])
    s = t2ds(s_t)
    if hasLeftEdge:
        return leftEdge + '.' + s
    return s

# def randomPrefix(l, alphabet):
#     return randomString(alphabet, l, hasLeftEdge=True)

def randomPrefixFromTriphones(triphones, l, hasLeftEdge=True):
    def foo(triphonesSoFar, max_length):
        s = threeFactorSequenceToDS(triphonesSoFar)
        s_t = ds2t(s)
        l = len(s_t)
        if l == max_length:
            return s

        rightmost_symbol = s_t[-1]
        triphonesBeginningWithRMS = {t for t in triphones if ds2t(t)[0] == rightmost_symbol}
        if l + 2 == max_length:
            wordFinalTriphones = list({t for t in triphonesBeginningWithRMS if ds2t(t)[2] == rightEdge})
            triphonesToChooseFrom = wordFinalTriphones
        else:
            wordMedialTriphones = list({t for t in triphonesBeginningWithRMS if ds2t(t)[2] != rightEdge})
            triphonesToChooseFrom = wordMedialTriphones
        nextTriphone = choice(triphonesToChooseFrom)
        triphonesSoFar.append(nextTriphone)
        return foo(triphonesSoFar, max_length)
    if hasLeftEdge:
        wordInitialTriphones = list({t for t in triphones if ds2t(t)[0] == leftEdge})
        return foo([choice(wordInitialTriphones)], max_length = l)
    else:
        raise Exception("Currently unsupported.")
#         return foo([choice(wordInitialTriphones)

def replaceXj(s, j, x):
    s_t = ds2t(s)
    s_l = list(s_t)
    s_l[j] = x
    s_t = tuple(s_l)
    return t2ds(s_t)

def removeXj(s, j):
    return replaceXj(s, j, '_')

def removeXi(x0k):
    l = len(ds2t(x0k))
    return removeXj(x0k, l-2)

def getPrefixes(s):
    if type(s) == str:
        sAsTup = ds2t(s)
    elif type(s) == tuple:
        sAsTup = s
    else:
        raise Exception('s must be a string or a tuple.')
    prefsAsTuples = set(sAsTup[0:i] for i in range(1, len(sAsTup)+1))
    return set(map(t2ds, prefsAsTuples))

def getProperPrefixes(s):
    Ps = getPrefixes(s)
    return {p for p in Ps if p[-1] != rightEdge}

def isProperPrefix(word, prefix):
    PPs = getProperPrefixes(word)
    return prefix in PPs

def hasAsPrefix(word, prefix):
    if type(prefix) == str:
        prefix_t = ds2t(prefix)
    elif type(prefix) == tuple:
        prefix_t = prefix
    else:
        raise Exception('prefix should be a dotted string or a tuple.')
    if type(word) == str:
        word_t = ds2t(word)
    elif type(word) == tuple:
        word_t = word
    else:
        raise Exception('word should be a dotted string or a tuple.')
    
    l = len(prefix_t)
    return word_t[0:l] == prefix_t



def d_s(x, y):
    '''
    Hamming distance between symbol x and symbol y.
    '''
    return x != y

def d_h(u, v):
    '''
    Hamming distance between strings u and v.
    '''
    u_t = ds2t(u)
    v_t = ds2t(v)
    if len(u_t) != len(v_t):
        return np.infty
    return sum(tuple(starmap(d_s, zip(u_t,v_t))))

def hamming_neighbors(s, W):
    '''
    Returns the strings of W that are exactly Hamming distance 1 from s.
    '''
    return h_sphere(1, s, W)

def h_sphere(k, s, W, exclude_s = False):
    '''
    Returns the strings of W that are exactly Hamming distance k from s.
    '''
    sphere = {v for v in W if d_h(s,v) == k}
    if exclude_s:
        return sphere - {s}
    return sphere

def h_neighborhood(k, s, W, exclude_s = False):
    '''
    Returns all strings of W whose Hamming distance from s is <= k.
    '''
    N = {v for v in W if d_h(s,v) <= k}
    if exclude_s:
        return N - {s}
    return N

def getSpheres(s, W):
    '''
    Returns a mapping from [0,len(s)-1] to the corresponding 
    Hamming spheres of s in W.
    '''
    D = range(len(ds2t(s)))
    spheres = {d:h_sphere(d, s, W) for d in D}
    return spheres

def neighborhood_measures(k, s, W, M, exclude_s = False):
    '''
    Applies a measure M (dictionary) to each of the k-neighbors
    of s in W.
    '''
    N = h_neighborhood(k, s, W, exclude_s)
    Ms = {v:M[v] for v in N}
    return Ms



# from os.path import isfile
def exists(fname):
    return isfile(fname)

def importSeqs(seq_fn):
    phoneSeqsAsStr = []
    with open(seq_fn, 'r') as the_file:
        for row in the_file:
            phoneSeqsAsStr.append(row.rstrip('\r\n'))
    return set(phoneSeqsAsStr)

def exportSeqs(seq_fn, seqs):
    with open(seq_fn, 'w') as the_file:
        for seq in seqs:
            the_file.write(seq + '\n')

def exportDict(fn, d):
    with codecs.open(fn, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii = False, indent = 4)
        
def importDict(fn):
    with open(fn, encoding='utf-8') as data_file:
        d_in = json.loads(data_file.read())
    return d_in

def castValuesToSets(d):
    return {k:set(d[k]) for k in d}

def castSetValuesToTuples(d):
    return {k:tuple(d[k]) for k in d}

diphone_analyses = ('destressed stimuli', 'stressed stimuli', 'destressed response')

def importNphoneAnalysis(N, which_align):
    assert which_align in {'unaligned', 'Hammond-aligned', 'IPhOD-aligned'}
    assert N in {1,2,3}

    which_infix = {1:'',
                   2:'',
                   3:'diphone-based'}[N]
    which_suffix = {1:{'licit':'',
                       'illicit':''},
                    2:{'licit':'',
                       'illicit':'illegal'},
                    3:{'licit':'constructible',
                       'illicit':'illegal'}}[N]
    which_n = {1:'uniphones',
               2:'diphones',
               3:'triphones'}[N]
    file_ext = '.txt'

    which_licit = {1:('licit',),
                   2:('licit', 'illicit'),
                   3:('licit', 'illicit')}[N]

    which_stress_which_diph = diphone_analyses

    analysis = dict()
    for each_licit in which_licit:
#         print('each_licit = {0}'.format(each_licit))
        analysis[each_licit] = dict()
        for each_stress_each_diph in which_stress_which_diph:
#             print('each_stress_each_diph = {0}'.format(each_stress_each_diph))
            my_suff = ' '.join([each for each in [each_stress_each_diph, which_infix, which_suffix[each_licit], which_n] if each != ''])
            analysis_fn = which_align + '_' + my_suff + file_ext
#             analysis_fn = which_align + '_' + ' '.join([each_stress_each_diph, which_infix, which_suffix[each_licit], which_n]) + file_ext
            print('Importing: ' + analysis_fn)
            analysis[each_licit][each_stress_each_diph] = importSeqs(analysis_fn)
    return analysis



def rev(t):
    return tuple(reversed(t))

def seqsToIndexMap(seqs):
    sorted_seqs = sorted(seqs)
    myIndexMap = dict(map(rev, enumerate(sorted_seqs)))
    return myIndexMap

def indexToSeqMap(seqs):
    sorted_seqs = sorted(seqs)
    mySeqMap = dict(enumerate(sorted_seqs))
    return mySeqMap

def areInverses(dictA, dictB):
    return all(dictB[dictA[k]] == k for k in dictA)

def seqMapToOneHots(seqMap):
    n = len(seqMap.keys())
    one_hots = np.zeros((n,n))
    for (seq, idx) in seqMap.items():
        one_hots[idx][idx] = 1.0
    return one_hots

def seqToOneHot(seq, seqMap, one_hots):
    return one_hots[seqMap[seq]]

def seqsToOneHotMap(seqs):
    seqMap = seqsToIndexMap(seqs)
    one_hots = seqMapToOneHots(seqMap)
    return {seq:one_hots[seqMap[seq]]
            for seq in seqMap}

def oneHotToSeqMap(seqs):
    sorted_seqs = sorted(seqs)
    seqsToOHmap = seqsToOneHotMap(seqs)
    seqsToOHmap_t = mapValues(tuple, seqsToOHmap)
    OHtoSeqs = dict(map(rev, seqsToOHmap_t.items()))
    def OHtoSeq(OH_vector):
        OH_t = tuple(OH_vector)
        return OHtoSeqs[OH_t]
    return OHtoSeq



# Parallel dictionary definition and data processing w/ progress reports


# from time import localtime, strftime
def stamp():
    return strftime('%H:%M:%S', localtime())

def stampedNote(note):
    print('{0} @ {1}'.format(note, stamp()))

def startNote(note):
    stampedNote('Start ' + note)
    
def endNote(note):
    stampedNote('End ' + note)

def processDataWProgressUpdates(f, data):
    print('Start @ {0}'.format(stamp()))
    l = len(data)
    benchmarkPercentages = [1,2,3,5,10,20,30,40,50,60,70,80,90,95,96,97,98,99,100]
    benchmarkIndices = [round(each/100.0 * l) for each in benchmarkPercentages]
    for i, d in enumerate(data):
        if i in benchmarkIndices:
            print('{0} | {0}/{1} = {2} | {3} | {4}'.format(i, l, i/l, d, stamp()))
        f(d)
    print('Finish @ {0}'.format(stamp()))
        
def constructDictWProgressUpdates(f, data, a_dict):
    def g(d):
        a_dict.update({d:f(d)})
    processDataWProgressUpdates(g, data)
    
def parallelDictDefinition(f, data, jobs, backend="multiprocessing", verbosity=5):
    def g(d):
        return (d, f(d))
    return dict( Parallel(n_jobs=jobs, backend=backend, verbose=verbosity)(delayed(g)(d) for d in data) )


