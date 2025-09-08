(* ::Package:: *)

(* ::Title:: *)
(*Cusp Rule Utilities for f-Graphs \[LongDash] Refactored*)


(* ::Section:: *)
(*Public Symbols & Usage*)


niceTime::usage                    = "niceTime[seconds] -> \"h:mm:ss\"-style string for progress messages.";
mapProgress::usage                 = "mapProgress[f, list] maps with a compact progress indicator.";
littleGroup::usage                 = "littleGroup[expr] returns the multiset of x[i,j] factors in numerator/denominator with signed multiplicities.";

loadPlanarGraphDials::usage        = "loadPlanarGraphDials[loop, dataDir] -> list of dials for that loop.";
loadPlanarGraphEdges::usage        = "loadPlanarGraphEdges[loop, dataDir] -> edge list for that loop with vertices as pairs.";
loadFGraphNums::usage              = "loadFGraphNums[loop, dataDir] -> list of numerator monomials (as Times of x[i,j]).";
fGraphList::usage                  = "fGraphList[loop, dataDir] -> list of f-graphs as (Times[numerator]/Times[denominator]).";
amplitudeCoefficients::usage       = "amplitudeCoefficients[loop, dataDir] -> vector of coefficients for f-graphs.";

fnumToDialnum::usage               = "fnumToDialnum[loop, fnum, dataDir] maps a flat index to {dialIndex, withinDialIndex}.";
dialnumToFnum::usage               = "dialnumToFnum[loop, {dial, idx}, dataDir] inverse mapping.";

fgraphToGraph::usage               = "fgraphToGraph[f] -> Graph of denominator edges.";
graphIsomorphisms::usage           = "graphIsomorphisms[f1, f2] -> list of base-label permutations mapping denominator+numerator.";
graphIsomorphism::usage            = "graphIsomorphism[f1, f2] -> one isomorphism (or {}).";
graphEquivalentQ::usage            = "graphEquivalentQ[f1, f2] -> True if equivalent under relabeling.";
distinctGraphs::usage              = "distinctGraphs[list] -> de-duplicate by f-graph equivalence.";
graphAutomorphisms::usage          = "graphAutomorphisms[f] -> automorphisms of denominator graph.";
graphSymmetryFactor::usage         = "graphSymmetryFactor[f] -> |Aut(G_den)|.";
canonicalizeGraph::usage           = "canonicalizeGraph[f] -> f with denominator canonically labeled and numerator edges ordered.";
canonicalizeFGraph::usage          = "canonicalizeFGraph[f] canonicalizes both denominator and numerator using automorphisms.";

orderedVertsToFaces::usage         = "orderedVertsToFaces[dials] -> faces as cycles given cyclic order at vertices.";
cycle::usage                       = "cycle[list, v] rotates list so it starts at v.";
myComplement::usage                = "myComplement[full, todel] deletes a single occurrence (multiset complement).";

detectDoubleTriangles::usage       = "detectDoubleTriangles[{loop, fgn}, dataDir] -> list of double-triangle pairs of faces.";
detectNonIsoDoubleTriangles::usage = "detectNonIsoDoubleTriangles[{loop, fgn}, dataDir] -> representatives modulo automorphisms.";
doubleTriangleToVWV1V2::usage      = "doubleTriangleToVWV1V2[{{...},{...}}] -> {v,w,v1,v2} in a fixed orientation.";
allSidewaysRelationsToFGraph::usage= "allSidewaysRelationsToFGraph[{loop,fgn}, dataDir] -> list of {coeffsL, coeffR, {loop,fgs},{loop-1,fgShrink}} for cusp-rule neighbors.";

allRelatedFGs::usage               = "allRelatedFGs[{loop,fgn},{v,w,v1,v2}, dataDir] \[LongDash] low-level cusp-rule relation for one oriented double triangle.";
highlightGraph::usage              = "highlightGraph[{loop,fgn}, dt, dataDir] -> Column[HighlightGraph[..., ...], numerator].";
highlightAllDoubleTriangles::usage = "highlightAllDoubleTriangles[{loop,fgn}, dataDir] -> list of highlights.";


(* ::Section:: *)
(*Options & Utilities*)


(* Compact time pretty-printer for progress bars *)
niceTime[timeInSec_] := Module[{t = N@timeInSec, q},
  Which[
    !NumericQ[t] || t <= 0, "",
    t >= 24*3600, q = QuotientRemainder[t, 24*3600]; ToString[q[[1]]] <> " d " <> DateString[q[[2]], {"Hour", ":", "Minute"}],
    True, DateString[t, {"Hour", ":", "Minute", ":", "Second"}]
  ]
];

(* Simple progress mapping (keeps your old style but lightweight) *)
mapProgress[f_, list_List] := Module[{n = Length[list], i = 0, t0 = AbsoluteTime[]},
  Monitor[
    Map[(i++; f[#]) &, list],
    Column[{
      Row[{"Progress: ", i, "/", n}],
      ProgressIndicator[i/n],
      If[i > 0, Row[{"ETA: ", niceTime[(AbsoluteTime[] - t0)*(n - i)/i]}], ""]
    }]
  ]
];

(* Signed multiset of x[i,j] exponents in an expression *)
littleGroup[graph_] := Module[{num, den, heads, baseLabels},
  num = Numerator[graph];
  den = Denominator[graph];
  heads = Cases[graph, x[a_, b_], {0, Infinity}];
  baseLabels = Sort@DeleteDuplicates[heads];
  With[
    {
      eNum = Exponent[num, #] & /@ baseLabels,
      eDen = Exponent[den, #] & /@ baseLabels
    },
    Thread[baseLabels -> (eNum - eDen)]
  ]
];



(* ::Section:: *)
(*File-backed data loaders (pure, memorized)*)


ClearAll[loadPlanarGraphDials, loadPlanarGraphEdges, loadFGraphNums, fGraphList, amplitudeCoefficients];
SetAttributes[#, HoldAll] & /@ {loadPlanarGraphDials, loadPlanarGraphEdges, loadFGraphNums, fGraphList, amplitudeCoefficients};

loadPlanarGraphDials[loop_Integer, dataDir_String] := loadPlanarGraphDials[loop, dataDir] =
  Module[{file = FileNameJoin[{dataDir, "fGraph_dials_" <> ToString[loop] <> ".tex"}]},
    If[FileExistsQ[file], ReadList[file], {{}}]
  ];

loadPlanarGraphEdges[loop_Integer, dataDir_String] := loadPlanarGraphEdges[loop, dataDir] =
  Module[{file = FileNameJoin[{dataDir, "fGraph_edges_" <> ToString[loop] <> ".tex"}], raw, rule},
    If[FileExistsQ[file],
      raw = ReadList[file];
      rule = Thread[Range[Binomial[4 + loop, 2]] -> Subsets[Range[4 + loop], {2}]];
      raw /. rule,
      {{}}
    ]
  ];

loadFGraphNums[loop_Integer, dataDir_String] := loadFGraphNums[loop, dataDir] =
  Module[{file = FileNameJoin[{dataDir, "fGraph_nums_" <> ToString[loop] <> ".tex"}], raw, rule},
    If[FileExistsQ[file],
      raw = ReadList[file];
      rule = Thread[Range[Binomial[4 + loop, 2]] -> Subsets[Range[4 + loop], {2}]];
      raw = Times @@@ (x @@@ # & /@ # & /@ (raw /. rule));
      raw,
      {{}}
    ]
  ];

fGraphList[loop_Integer, dataDir_String] := fGraphList[loop, dataDir] =
  Module[{edges = loadPlanarGraphEdges[loop, dataDir], nums = loadFGraphNums[loop, dataDir]},
    If[edges === {{}} || nums === {{}}, {{}},
      With[{den = Times @@@ (x @@@ # & /@ edges)},
        Join @@ (nums (1/den))
      ]
    ]
  ];

amplitudeCoefficients[loop_Integer, dataDir_String] := amplitudeCoefficients[loop, dataDir] =
  Module[{file = FileNameJoin[{dataDir, "amplitudeCoefficients_" <> ToString[loop] <> ".tex"}]},
    If[FileExistsQ[file], Get[file], {}]
  ];

(* Mapping indices between the flat list and per-dial blocks *)
fnumToDialnum[loop_Integer, fnum_Integer, dataDir_String] :=
 Module[{blocks = Accumulate[Length /@ loadFGraphNums[loop, dataDir]]},
  With[{pos = First@First@Position[Table[Range[1 + If[i == 1, 0, blocks[[i - 1]]], blocks[[i]]], {i, Length[blocks]}], fnum]},
    {pos, fnum - If[pos == 1, 0, blocks[[pos - 1]]]}
  ]
];

dialnumToFnum[loop_Integer, {dial_Integer, idx_Integer}, dataDir_String] :=
 Module[{blocks = Accumulate[Length /@ loadFGraphNums[loop, dataDir]]},
  If[dial == 1, idx, blocks[[dial - 1]] + idx]
];



(* ::Section:: *)
(*Graph helpers*)


(* Denominator graph from an f-graph rational function *)
fgraphToGraph[f_] := Graph[(List @@ Denominator[f]) /. x -> List, GraphLayout -> "PlanarEmbedding"];

(* A tiny safe TakeSmallest wrapper (v13+ has TakeSmallest[..]) *)
takeSmallest[list_, p_Integer] := Take[Sort[list], p];

(* Numerator monomial to counted list of x[i,j]^power *)
numeratorMonomialToList[num_] := Module[{heads, exps},
  heads = Cases[num, x[__, __], Infinity];
  exps = Exponent[num, #] & /@ DeleteDuplicates[heads];
  Flatten@MapThread[ConstantArray, {DeleteDuplicates[heads], exps}]
];

(* Canonicalization that first canonicalizes denominator graph and then orbits numerator under Aut(G) *)
canonicalizeFGraph[f_] := Module[
  {denG = fgraphToGraph[f], denGcan, iso, numList, numOrbit, denCan, firstNum},
  denGcan = CanonicalGraph[denG];
  iso = First@Normal@FindGraphIsomorphism[denG, denGcan];
  numList = numeratorMonomialToList@Numerator[f] /. iso /. x[a_, b_] :> x @@ Sort[{a, b}];
  denCan  = Times @@ (EdgeList[denGcan] /. UndirectedEdge -> x);
  numOrbit = (Times @@@ PermutationReplace[numList, #]) & /@ GraphAutomorphismGroup[denGcan];
  firstNum = First@takeSmallest[numOrbit /. x[exp__] :> x @@ Sort[{exp}], 1];
  firstNum/denCan
];


graphIsomorphisms[fcnList__]/;(Length[{fcnList}]==2):=Block[{factors={fcnList},isoList,bases,nums,baseLabels=DeleteDuplicates[Cases[{fcnList}[[1]],x[y__]:>y,{0,\[Infinity]}]]},
If[Length[DeleteDuplicates[factors]]==1,
	((bases=(Function[{q},List@@@DeleteDuplicates[Cases[Denominator[q],x[y__],{0,\[Infinity]}]]]/@factors);
	nums=(factors*((Times@@(x@@@#))&/@bases));
	bases=Graph[UndirectedEdge@@@#]&/@bases;
	isoList=(FindGraphIsomorphism[#1,#2,All]&@@bases);
	If[Length[isoList]==0,{},(isoList=(Function[{iso},Thread[Rule@@(baseLabels/.{{},iso})]]/@isoList);
	isoList=Select[isoList,SameQ[nums[[2]],(nums[[1]]/.{x[y__]:>(x@@Sort[{y}/.#])})]&])])),(Which[Not[SameQ@@(LeafCount/@factors)],{},True,(bases=(Function[{q},List@@@DeleteDuplicates[Cases[Denominator[q],x[y__],{0,\[Infinity]}]]]/@factors);nums=(factors*((Times@@(x@@@#))&/@bases));
	bases=Graph[UndirectedEdge@@@#]&/@bases;isoList=(FindGraphIsomorphism[#1,#2,All]&@@bases);If[Length[isoList]==0,{},(isoList=(Function[{iso},Thread[Rule@@(baseLabels/.{{},iso})]]/@isoList);
	isoList=Select[isoList,SameQ[nums[[2]],(nums[[1]]/.{x[y__]:>(x@@Sort[{y}/.#])})]&])])])]];

graphIsomorphism[f1_, f2_] /; Length[{f1, f2}] == 2 :=
  Module[{isos = graphIsomorphisms[f1, f2]}, If[isos === {}, {}, First@isos]];

graphEquivalentQ[f1_, f2_] := graphIsomorphism[f1, f2] =!= {};

distinctGraphs[seed_List] := DeleteDuplicates[seed, graphEquivalentQ];

graphAutomorphisms[f_] := graphIsomorphisms[f, f];
graphSymmetryFactor[f_] := Length@graphAutomorphisms[f];

(* Canonicalize denominator labels only; keep for debugging *)
canonicalizeGraph[f_] := Module[{edges, base, auto},
  edges = List @@@ DeleteDuplicates@Cases[Denominator[f], x[y__], {0, Infinity}];
  base  = Graph[UndirectedEdge @@@ edges];
  auto  = (Thread[Rule @@ (DeleteDuplicates@Flatten[edges] /. {{}, #})] & /@ FindGraphIsomorphism[base, CanonicalGraph@base])[[1]];
  f /. x[q__] :> x @@ (Sort[{q} /. auto])
];


(* ::Section:: *)
(*Faces & cyclic helpers*)


cycle[list_List, v_] := Module[{p = FirstPosition[list, v, Missing[]]},
  If[p === Missing[], list, RotateLeft[list, p[[1]] - 1]]
];

orderedVertsToFaces[dials_List] := Module[{edgeRules, fullEdgeList},
  edgeRules = Join @@ Table[(Rule[{#1, j}, {j, #2}] & @@@ Partition[dials[[j]], 2, 1, 1]), {j, Length[dials]}];
  fullEdgeList = edgeRules[[All, 1]];
  Sort@DeleteDuplicates[
    Function[{start},
      With[{term = DeleteDuplicates@NestWhile[
         Append[#, (#[[-1]] /. edgeRules)] &,
         {start},
         Length@DeleteDuplicates[#] == Length[#] &
       ]},
        Sort[Partition[Reverse@term, Length[term], 1, 1]][[1]]
      ]
    ] /@ fullEdgeList
  ]
];

myComplement[full_List, todel_List] := Fold[Delete[#1, First@Position[#1, #2, 1, 1]] &, full, todel];



(* ::Section:: *)
(*Cusp-rule core building blocks (decomposed)*)


(* Detect all double triangles (pairs of triangles sharing two vertices) *)
detectDoubleTriangles[{loop_Integer, fgn_Integer}, dataDir_String] := Module[
  {dials, triangles, pairs},
  dials      = loadPlanarGraphDials[loop, dataDir][[fnumToDialnum[loop, fgn, dataDir][[1]]]];
  triangles  = Select[orderedVertsToFaces[dials], Length[#] == 3 &];
  pairs      = Select[Subsets[triangles, {2}], Length[Intersection @@ #] == 2 &];
  pairs
];

(* Mod out by denominator automorphisms to avoid duplicates *)
detectNonIsoDoubleTriangles[idx : {loop_Integer, fgn_Integer}, dataDir_String] := Module[
  {pairs = detectDoubleTriangles[idx, dataDir], autos, current, noniso = {}},
  autos = graphAutomorphisms[fGraphList[loop, dataDir][[fgn]]];
  current = pairs;
  While[Length[current] > 0,
    AppendTo[noniso, current[[1]]];
    current = Complement[
      current,
      Table[current[[1]] /. \[Sigma], {\[Sigma], autos}],
      SameTest -> ((Sort /@ {#1[[1]], #1[[2]]} == Sort /@ {#2[[1]], #2[[2]]}) || (Sort /@ {#1[[2]], #1[[1]]} == Sort /@ {#2[[1]], #2[[2]]}) &)
    ]
  ];
  noniso
];

(* Orient a double triangle into {v,w,v1,v2} (v,w are 3-valent top/bottom; v1,v2 are 2-valent left/right) *)
doubleTriangleToVWV1V2[dt : {{__Integer} ..}] := Module[
  {tallied = Tally@Flatten@dt, vw, dtcyc, v, w, v1, v2},
  vw   = (Select[tallied, #[[2]] == 2 &][[All, 1]]);
  dtcyc = {cycle[dt[[1]], vw[[1]]], cycle[dt[[2]], vw[[1]]]};
  If[dtcyc[[1, 2]] =!= vw[[2]],
    {v, w, v1, v2} = {vw[[1]], vw[[2]], dtcyc[[2, 3]], dtcyc[[1, 2]]},
    {v, w, v1, v2} = {vw[[1]], vw[[2]], dtcyc[[1, 3]], dtcyc[[2, 2]]}
  ];
  {v, w, v1, v2}
];

(* Low-level cusp relation for one oriented double triangle *)
allRelatedFGs[{loop_Integer, fgn_Integer}, {v_, w_, v1_, v2_}, dataDir_String] :=
 Module[
  {
    n = loop + 4, f = fGraphList[loop, dataDir][[fgn]],
    dials, faces, facesv, facesw, upperfaces, lowerfaces, upperbig, lowerbig,
    newNumPtsUp, newNumPtsLow, newEdges, newDen, existingNum,
    existingList, toV, toW, remainNum, allPts, numeratorList, fgraphs, fcan,
    fgnums1, fgnums, shrunk, fgShrinkNum, coeffL, coeffR
  },
  dials      = loadPlanarGraphDials[loop, dataDir][[fnumToDialnum[loop, fgn, dataDir][[1]]]];
  faces      = orderedVertsToFaces[dials];
  facesv     = cycle[#, v] & /@ Select[faces, MemberQ[#, v] &];
  facesw     = cycle[#, w] & /@ Select[faces, MemberQ[#, w] &];

  (* Faces above and below the waist *)
  upperfaces = Drop[Table[Select[facesv, #[[2]] == vn &][[1]], {vn, cycle[dials[[v]], v1]}], -2];
  lowerfaces = Drop[Table[Select[facesw, #[[2]] == wn &][[1]], {wn, cycle[dials[[w]], v2]}], -2];

  upperbig   = Join[{v, v1}, Flatten[Drop[#, 2] & /@ upperfaces]];
  lowerbig   = Join[{w, v2}, Flatten[Drop[#, 2] & /@ lowerfaces]];

  newNumPtsUp  = Select[upperbig, Not@MemberQ[Join[dials[[v]], dials[[w]]], #] &];
  newNumPtsLow = Select[lowerbig, Not@MemberQ[Join[dials[[v]], dials[[w]]], #] &];

  newEdges   = Join[(x[v, #] & /@ newNumPtsUp), (x[w, #] & /@ newNumPtsLow)];
  newDen     = Denominator[f] Times @@ newEdges;

  existingNum    = loadFGraphNums[loop, dataDir][[Sequence @@ fnumToDialnum[loop, fgn, dataDir]]];
  existingList   = (List @@ existingNum /. Times -> List) /. a_^b_. :> ConstantArray[a, b] // Flatten;

  toV = Join[Cases[existingList, x[v, a_] :> a, Infinity], Cases[existingList, x[a_, v] :> a, Infinity]];
  toW = Join[Cases[existingList, x[w, a_] :> a, Infinity], Cases[existingList, x[a_, w] :> a, Infinity]];

  remainNum = Times @@ Select[existingList, Not@MemberQ[#, v] && Not@MemberQ[#, w] &];

  allPts = Join[newNumPtsUp, newNumPtsLow, toV, toW];

  numeratorList =
    DeleteDuplicates[
      (Product[x[v, i], {i, #}] Product[x[w, i], {i, myComplement[allPts, #]}]) & /@
        Subsets[allPts, {Length[newNumPtsUp] + Length[toV]}] /. x[a_, b_] :> x @@ Sort[{a, b}]
    ];

  fgraphs = (numeratorList remainNum)/newDen /. x[a_, b_] :> x @@ Sort[{a, b}];
  fcan    = canonicalizeFGraph /@ fgraphs;

  fgnums1 = Position[fGraphList[loop, dataDir] // (canonicalizeFGraph /@ #) &, #] & /@ fcan;
  fgnums  = Flatten[If[# === {}, Message[allRelatedFGs::miss, loop, fgn, {v, w, v1, v2}]; {}, #] & /@ fgnums1];

  shrunk = (fgraphs /. x[a_, b_] :> x[a /. {w -> v}, b /. {w -> v}]) (x[v, v] x[v1, v] x[v2, v])/x[v1, v2] /. x[a_, b_] :> x @@ Sort@{a /. {n -> w}, b /. {n -> w}};
  If[Length@Union[shrunk] =!= 1, Print["Different shrunk graphs detected in cusp operation."]];
  fgShrinkNum = With[{cf = canonicalizeFGraph@First@shrunk}, Quiet@Check[First@First@Position[fGraphList[loop - 1, dataDir] // (canonicalizeFGraph /@ #) &, cf], 0]];

  coeffR = If[fgShrinkNum === 0, 0, amplitudeCoefficients[loop - 1, dataDir][[fgShrinkNum]]];
  coeffL = amplitudeCoefficients[loop, dataDir][[fgnums]];

  If[Total[coeffL] =!= coeffR, Print["Not matching the cusp equation: ", {coeffL, coeffR, {loop, fgnums}, {loop - 1, fgShrinkNum}}]];

  {coeffL, coeffR, {loop, fgnums}, {loop - 1, fgShrinkNum}}
];
allRelatedFGs::miss = "Could not identify some related f-graphs at loop=`1`, fgn=`2`, dt=`3`.";

(* User-level: compute all sideways relations from non-isomorphic DTs *)
allSidewaysRelationsToFGraph[idx : {loop_Integer, fgn_Integer}, dataDir_String] :=
 Module[{dts = detectNonIsoDoubleTriangles[idx, dataDir]},
  allRelatedFGs[idx, doubleTriangleToVWV1V2[#], dataDir] & /@ dts
];



(* ::Section:: *)
(*Visualization helpers*)


highlightGraph[{loop_Integer, fgn_Integer}, dt : {{__Integer} ..}, dataDir_String] := Module[
  {G = PlanarGraph[fgraphToGraph[fGraphList[loop, dataDir][[fgn]]], VertexLabels -> "Name"], E},
  E = UndirectedEdge @@@ Join[Subsets[dt[[1]], {2}], Subsets[dt[[2]], {2}]];
  Column[{HighlightGraph[G, E], Numerator[fGraphList[loop, dataDir][[fgn]]]}]
];

highlightAllDoubleTriangles[idx : {loop_Integer, fgn_Integer}, dataDir_String] :=
  highlightGraph[idx, #, dataDir] & /@ detectNonIsoDoubleTriangles[idx, dataDir];



(* ::Section:: *)
(*Examples*)


(* Example 1: Canonicalization and equivalence on two explicit f-graphs (no files needed) *)
Module[{g1, g2},
  g1 = (x[4, 9] x[5, 6] x[7, 8] x[10, 11]^2)/
       (x[1, 6] x[1, 7] x[1, 8] x[1, 9] x[2, 3] x[2, 4] x[2, 10] x[2, 11] x[3, 5] x[3, 10] x[3, 11] x[4, 6] x[4, 8] x[4, 10] x[4, 11] x[5, 7] x[5, 9] x[5, 10] x[5, 11] x[6, 7] x[6, 8] x[6, 10] x[7, 9] x[7, 10] x[8, 9] x[8, 11] x[9, 11]);
  g2 = (x[4, 5] x[6, 9] x[7, 8] x[10, 11]^2)/
       (x[1, 4] x[1, 5] x[1, 6] x[1, 7] x[2, 3] x[2, 9] x[2, 10] x[2, 11] x[3, 8] x[3, 10] x[3, 11] x[4, 6] x[4, 7] x[4, 9] x[4, 11] x[5, 6] x[5, 7] x[5, 8] x[5, 10] x[6, 8] x[6, 11] x[7, 9] x[7, 10] x[8, 10] x[8, 11] x[9, 10] x[9, 11]);

  {
    "EquivalentQ" -> graphEquivalentQ[g1, g2],
    "CanonicalEqualQ" -> (Apply[Equal, canonicalizeFGraph /@ {g1, g2}] // Simplify)
  }
]

(* Example 2: Faces and double-triangle decanonicalizeFGraphtection (requires dataDir with your .tex files)
   Uncomment and set dataDir to your folder:

   dataDir = "/home/gabriele/Documents/GitHub/ML-correlator/consolidated_data";
   detectNonIsoDoubleTriangles[{4, 1}, dataDir]
   highlightAllDoubleTriangles[{4, 1}, dataDir]
*)

(* Example 3: Cusp-rule neighbors for one f-graph (file-backed)
   allSidewaysRelationsToFGraph[{6, 33}, dataDir]
*)


