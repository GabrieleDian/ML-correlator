(* ::Package:: *)

(* ::Section::Closed:: *)
(*General-Purpose Functions*)


If[Head[$FrontEnd] === FrontEndObject,
	(* Running on Notebook/FrontEnd*)
  $FileDirectory = NotebookDirectory[];
  SetDirectory[NotebookDirectory[]],
  (*Running on cluster or in script.*)
  $FileDirectory = Directory[]
];
Print["The file directory is: ", $FileDirectory];


 $DataPath = StringDrop[DirectoryName[$FileDirectory],-1];


fourPointDataDir= $DataPath<>"/consolidated_data";
Print["The data directory is: ", fourPointDataDir];


niceTime[timeInSec_]:=If[timeInSec<($TimeUnit/100)||Not[NumericQ[timeInSec]]||Precision[timeInSec]==0,"",Block[{measure=Select[Transpose[{(Quotient[Mod[timeInSec,#1],#2]&@@@Partition[{timeInSec 10,3.15569277216`*^7,3600*24.,3600.,60.,1.,10^-3,10.^-6,10.^-9},2,1]),{" years"," days"," hours"," minutes"," seconds"," ms"," \[Mu]s"," ns"}}],#[[1]]>0&]},If[Length[measure]>0,Row[Row[#,""]&/@measure[[1;;Min[2,Length[measure]]]],", "],""]]];
map[function_,list_]:=If[Length[list]>0,Module[{monitor=0,len=Length[list],newFcn,t00=AbsoluteTime[]},newFcn[i_]:=(monitor=i;function[list[[i]]]);Monitor[Map[newFcn,Range[len]],(Column[{ProgressIndicator[(monitor-1)/len,ImageSize->{1250,30},ImageMargins->0,BaselinePosition->Center],If[monitor>=1,Row[{If[monitor>1,Row[{niceTime[Round[AbsoluteTime[]-t00]]," so far; approx. ",niceTime[Round[((AbsoluteTime[]-t00)/(monitor-1))*(len-monitor+1)]]," remaining."}],"                                              "]," (",monitor-1,"/",len,")"}],""]},Alignment->Left,Spacings->0.1])]],Map[function,list]];
littleGroup[graph_]:=Block[{num=Numerator[graph],den=Denominator[graph]},num=Join@@(Function[{y,z},y&/@Range[z]]@@@FactorList[num][[2;;-1]]);den=Join@@(Function[{y,z},y&/@Range[z]]@@@FactorList[den][[2;;-1]]);-(Count[num,#,{0,\[Infinity]}]-Count[den,#,{0,\[Infinity]}])&/@Sort[DeleteDuplicates[Flatten[List@@@Cases[graph,_x,{0,\[Infinity]}]]]]];


(* ::Section::Closed:: *)
(*Basic Data Functions*)


planarGraphDials[loop_]:=Block[{fileName=FileNameJoin[{fourPointDataDir,(("fGraph_dials_"<>ToString[loop]<>".tex"))}]},If[FileExistsQ[fileName],Set[planarGraphDials[loop],ReadList[fileName]],{{}}]];
planarGraphEdges[loop_]:=Block[{fileName=FileNameJoin[{fourPointDataDir,(("fGraph_edges_"<>ToString[loop]<>".tex"))}],raw,rule},If[FileExistsQ[fileName],(raw=ReadList[fileName];rule=Thread[Rule[Range[Binomial[4+loop,2]],Subsets[Range[4+loop],{2}]]];Set[planarGraphEdges[loop],(raw/.rule)]),{{}}]];
fGraphNums[loop_]:=Block[{fileName=FileNameJoin[{fourPointDataDir,("fGraph_nums_"<>ToString[loop]<>".tex")}],raw,rule},If[FileExistsQ[fileName],(raw=ReadList[fileName];rule=Thread[Rule[Range[Binomial[4+loop,2]],Subsets[Range[4+loop],{2}]]];raw=Times@@@#&/@(x@@@#&/@#&/@(raw/.rule));Set[fGraphNums[loop],raw]),{{}}]];
fGraphList[loop_]:=If[FileExistsQ[FileNameJoin[{fourPointDataDir,("fGraph_nums_"<>ToString[loop]<>".tex")}]],Set[fGraphList[loop],(Join@@(fGraphNums[loop](1/(Times@@@(x@@@#&/@planarGraphEdges[loop])))))],{{}}];

amplitudeCoefficients[loop_]:=Block[{fileName=FileNameJoin[{fourPointDataDir,(("amplitudeCoefficients_"<>ToString[loop]<>".tex"))}]},If[FileExistsQ[fileName],Set[amplitudeCoefficients[loop],(<<(fileName))],{}]];


(* This takes you from the fnum list to the dialnumber list position *)
fnumtodialnum[loop_,fnum_]:=Position[Table[Range[#[[i]]+1,#[[i+1]]],{i,1,Length[#]-1}]&@Prepend[Accumulate[Length/@fGraphNums[loop]],0],fnum][[1]]
dialnumtofnum[loop_,{dial_,num_}]:=(Table[Range[#[[i]]+1,#[[i+1]]],{i,1,Length[#]-1}]&@Prepend[Accumulate[Length/@fGraphNums[loop]],0])[[dial,num]]


(* ::Section::Closed:: *)
(*Graph Analysis*)


takeSmallest[x_,p_] := Take[Sort[x],p]


graphIsomorphisms[fcnList__]/;(Length[{fcnList}]==2):=Block[{factors={fcnList},isoList,bases,nums,baseLabels=DeleteDuplicates[Cases[{fcnList}[[1]],x[y__]:>y,{0,\[Infinity]}]]},If[Length[DeleteDuplicates[factors]]==1,((bases=(Function[{q},List@@@DeleteDuplicates[Cases[Denominator[q],x[y__],{0,\[Infinity]}]]]/@factors);nums=(factors*((Times@@(x@@@#))&/@bases));bases=Graph[UndirectedEdge@@@#]&/@bases;isoList=(FindGraphIsomorphism[#1,#2,All]&@@bases);If[Length[isoList]==0,{},(isoList=(Function[{iso},Thread[Rule@@(baseLabels/.{{},iso})]]/@isoList);isoList=Select[isoList,SameQ[nums[[2]],(nums[[1]]/.{x[y__]:>(x@@Sort[{y}/.#])})]&])])),(Which[Not[SameQ@@(LeafCount/@factors)],{},True,(bases=(Function[{q},List@@@DeleteDuplicates[Cases[Denominator[q],x[y__],{0,\[Infinity]}]]]/@factors);nums=(factors*((Times@@(x@@@#))&/@bases));bases=Graph[UndirectedEdge@@@#]&/@bases;isoList=(FindGraphIsomorphism[#1,#2,All]&@@bases);If[Length[isoList]==0,{},(isoList=(Function[{iso},Thread[Rule@@(baseLabels/.{{},iso})]]/@isoList);isoList=Select[isoList,SameQ[nums[[2]],(nums[[1]]/.{x[y__]:>(x@@Sort[{y}/.#])})]&])])])]];
graphIsomorphism[fcnList__]/;(Length[{fcnList}]==2):=Block[{factors={fcnList},isoList,bases,nums,baseLabels=DeleteDuplicates[Cases[{fcnList}[[1]],x[y__]:>y,{0,\[Infinity]}]]},If[Length[DeleteDuplicates[factors]]==1,Thread[Rule[baseLabels,baseLabels]],(Which[Not[SameQ@@(LeafCount/@factors)],{},True,(bases=(Function[{q},List@@@DeleteDuplicates[Cases[Denominator[q],x[y__],{0,\[Infinity]}]]]/@factors);nums=(factors*((Times@@(x@@@#))&/@bases));bases=Graph[UndirectedEdge@@@#]&/@bases;isoList=(FindGraphIsomorphism[#1,#2,All]&@@bases);If[Length[isoList]==0,{},(isoList=(Function[{iso},Thread[Rule@@(baseLabels/.{{},iso})]]/@isoList);isoList=Select[isoList,SameQ[nums[[2]],(nums[[1]]/.{x[y__]:>(x@@Sort[{y}/.#])})]&,1])])])]];
graphEquivalentQ[fcnList__]/;(Length[{fcnList}]==2):=Not[graphIsomorphism[fcnList]==={}];
distinctGraphs[seedList_]:=DeleteDuplicates[seedList,graphEquivalentQ];
graphAutomorphisms[graph_]:=graphIsomorphisms[graph,graph];
graphSymmetryFactor[graph_]:=Length[graphAutomorphisms[graph]];
canonicalizeGraph[graph_]:=Block[{baseEdges=List@@@DeleteDuplicates[Cases[Denominator[graph],x[y__],{0,\[Infinity]}]],baseLabels,baseGraph,auto},baseLabels=DeleteDuplicates[Flatten[baseEdges]];baseGraph=Graph[UndirectedEdge@@@baseEdges];auto=(Function[{iso},Thread[Rule@@(baseLabels/.{{},iso})]]/@FindGraphIsomorphism[baseGraph,CanonicalGraph[baseGraph]])[[1]];graph/.{x[q__]:>x@@(Sort[{q}/.auto])}];

orderedVertsToFaces[dialList_]:=Block[{edgeRules,fullEdgeList},edgeRules=Join@@Table[(Rule[{#1,j},{j,#2}]&@@@Partition[dialList[[j]],2,1,1]),{j,Length[dialList]}];fullEdgeList=edgeRules[[All,1]];Sort@DeleteDuplicates[Function[{start},(Function[{term},Sort[Partition[Reverse@term,Length[term],1,1]][[1]]]@DeleteDuplicates[Join@@NestWhile[Append[#,(#[[-1]]/.edgeRules)]&,{start},Length[DeleteDuplicates[#]]==Length[#]&]])]/@fullEdgeList]];


orderedVertsToFaces[dialList_]:=Block[{edgeRules,fullEdgeList},edgeRules=Join@@Table[(Rule[{#1,j},{j,#2}]&@@@Partition[dialList[[j]],2,1,1]),{j,Length[dialList]}];fullEdgeList=edgeRules[[All,1]];Sort@DeleteDuplicates[Function[{start},(Function[{term},Sort[Partition[Reverse@term,Length[term],1,1]][[1]]]@DeleteDuplicates[Join@@NestWhile[Append[#,(#[[-1]]/.edgeRules)]&,{start},Length[DeleteDuplicates[#]]==Length[#]&]])]/@fullEdgeList]]


degree[fgraph_,v_]:=Count[Denominator[fgraph]/.Times->List/.x->List//Flatten,v]


fgraphtograph[fgraph_]:=Graph[(List@@Denominator[fgraph])/.x->List]


numeratormonomialtolist[numerator_]:=If[Head[numerator]===Integer,{},ConstantArray@@@Transpose[{Cases[#,x[_,_],Infinity],Exponent[#,Cases[#,x[_,_],Infinity]]}]&@({numerator/.Times->List}//Flatten)//Flatten]


canonicalizefgraph[fgraph_]:=Module[{fgraph2,dengraphcan,dengraphtocanisomorphism,dencan,numcan1,numcan,dengraph,numedgelist},fgraph2=fgraph/.x[a_,b_]:>x[b,a]/;(b<a);
dengraph=fgraphtograph[fgraph2];
dengraphcan=CanonicalGraph[dengraph];
dengraphtocanisomorphism=((FindGraphIsomorphism[dengraph,dengraphcan]//Normal)[[1]]);
numedgelist=numeratormonomialtolist[Numerator[fgraph2]];
numcan1=numedgelist/.dengraphtocanisomorphism/.x[a_,b_]:>x[b,a]/;(b<a);
dencan=Times@@(EdgeList[dengraphcan]/.UndirectedEdge->x);
numcan=takeSmallest[Times@@@PermutationReplace[numcan1,GraphAutomorphismGroup[dengraphcan]]/.x[exp__]:>x[Sequence@@Sort[{exp}]]/;(Sort[{exp}]=!={exp}),1][[1]];
numcan/dencan]


Clear[fGraphListcan]
fGraphListcan[loop_]:=fGraphListcan[loop]=canonicalizefgraph/@fGraphList[loop]


cycle[dial_,ve_]:=If[MemberQ[dial,ve],Join[Take[dial,{Position[dial,ve][[1,1]],Length[dial]}],Take[dial,{1,Position[dial,ve][[1,1]]-1}]],dial]


displayfgraph[fgraph_]:=Column[{PlanarGraph[Denominator[fgraph]/.Times->List/.x->List,VertexLabels->"Name"],Numerator[fgraph]}]


(* ::Section:: *)
(*generate new graphs via binary relations*)


Get["IGraphM`"]


detectnonisomorphicdoubletrianglesfgraph[fgraph_]:=Module[{alldts,dtscurrent,dtsnoniso,autos},alldts=FindIsomorphicSubgraph[(fgraph//displayfgraph)[[1,1]],Graph[{{1,2},{1,3},{2,3},{1,4},{3,4}}],All];
autos=graphAutomorphisms[fgraph];
dtscurrent=alldts;
dtsnoniso={};
While[Length[dtscurrent]>0,
dtsnoniso=Append[dtsnoniso,dtscurrent[[1]]];
dtscurrent=Complement[dtscurrent,Table[VertexReplace[dtscurrent[[1]],ii],{ii,autos}],SameTest->IGSameGraphQ  ]];
({#[[1,1]],#[[2,2]],#[[1,2]],#[[3,2]]}&)/@EdgeList/@dtsnoniso]


(*Join two list such that if they form a double triangle the shared edges correposnd to the first two position*)
joinTriangles[t1_,t2_]:=Join[Intersection[t1,t2],SymmetricDifference[t1,t2]]
(*Sort correctly the output of join Triangles to give a 4-cycle*)
SwapSecondThird[list_List] := ReplacePart[list, {2 -> list[[3]], 3 -> list[[2]]}]


doubleTrFromTriangles[triangles_]:=SwapSecondThird/@Select[(joinTriangles@@@Subsets[triangles,{2}]),Length[#]===4&]


CanonicalizeCyclic[list_List] := 
  MinimalBy[RotateLeft[list, #] & /@ Range[0, Length[list] - 1], Identity][[1]]
  
  CanonicalizeDihedral[list_List] := Module[
  {rots, reflRots},
  
  (* all cyclic rotations *)
  rots = Table[RotateLeft[list, k], {k, 0, Length[list] - 1}];
  
  (* all cyclic rotations of the reversed list *)
  reflRots = Table[RotateLeft[Reverse[list], k], {k, 0, Length[list] - 1}];
  
  (* pick lexicographically minimal among both sets *)
  MinimalBy[Join[rots, reflRots], Identity][[1]]
]


(*Test
CanonicalizeDihedral/@doubleTrFromTriangles[triangles]//Sort
CanonicalizeDihedral/@doubletrianglesfgraph[fg]//Sort
%===%%
*)


fullRungRule[fg_]:=Module[{facets,goodSquares,nn,newfgs},
facets=facetsFgraph[fg];
triangles=Select[facets,Length[#]===3&];
goodSquares=Join[Select[facets,Length[#]===4&],doubleTrFromTriangles[triangles]];
nn=Max[Cases[facets,_Integer,Infinity]];
newfgs=(fg x[#[[1]],#[[3]]]x[#[[2]],#[[4]]]/(x[#[[1]],nn+1]x[#[[2]],nn+1]x[#[[3]],nn+1]x[#[[4]],nn+1])/.x[a_,b_]:>x[b,a]/;(a>b))&/@goodSquares;
canonicalizefgraph/@newfgs]


facetsFgraph[fg_]:=PlanarFaceList[Graph[UndirectedEdge@@@(List@@Denominator[fg])]]


doubletrianglesfgraph[fgraph_]:=VertexList/@FindIsomorphicSubgraph[(fgraph//displayfgraph)[[1,1]],Graph[{{1,2},{1,3},{2,3},{1,4},{3,4}}],All]


myRungRule[fg_]:=Module[{dts,nn,newfgs},
dts=doubletrianglesfgraph[fg];
nn=Max[Cases[fg,_Integer,Infinity]];
newfgs=(fg x[#[[1]],#[[3]]]x[#[[2]],#[[4]]]/(x[#[[1]],nn+1]x[#[[2]],nn+1]x[#[[3]],nn+1]x[#[[4]],nn+1])/.x[a_,b_]:>x[b,a]/;(a>b))&/@dts;
canonicalizefgraph/@newfgs]


rungrulegenerate[fg_]:=Module[{dts,nn,newfgs},
dts=detectnonisomorphicdoubletrianglesfgraph[fg];
nn=Max[Cases[fg,_Integer,Infinity]];
newfgs=(fg x[#[[1]],#[[2]]]x[#[[3]],#[[4]]]/(x[#[[1]],nn+1]x[#[[2]],nn+1]x[#[[3]],nn+1]x[#[[4]],nn+1])/.x[a_,b_]:>x[b,a]/;(a>b))&/@dts;
canonicalizefgraph/@newfgs]


(* Equivalence test:
fGraphListcan[nn][[1]]
DeleteDuplicates[myRungRule[%]]
rungrulegenerate[%%]==%*)


(* ::Input:: *)
(*(* Output is list of vertices as {1,2,3,4,5,6,7,8} in . *)*)


detectnonisohexagonwithtwovertsinmiddle[fgraph_]:=Module[{alldts,dtscurrent,dtsnoniso,autos,orderedverts},alldts=FindIsomorphicSubgraph[(fgraph//displayfgraph)[[1,1]],Graph[{1, 2, 3, 4, 5, 6, 7, 8}, {Null, {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 1}, {1, 8}, {2, 8}, {3, 8}, {4, 8}, {4, 7}, {5, 7}, {6, 7}, {1, 7}, {7, 8}}}, {FormatType -> TraditionalForm, FormatType -> TraditionalForm, GraphLayout -> {"Dimension" -> 2}, VertexLabels -> {"Name"}}],All];
autos=graphAutomorphisms[fgraph];
dtscurrent=alldts;
dtsnoniso={};
While[Length[dtscurrent]>0,dtsnoniso=Append[dtsnoniso,dtscurrent[[1]]];dtscurrent=Complement[dtscurrent,Table[VertexReplace[dtscurrent[[1]],ii],{ii,autos}],SameTest->IGSameGraphQ  ]];
(Range[8]/.(FindGraphIsomorphism[Graph[{1, 2, 3, 4, 5, 6, 7, 8}, {Null, {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 1}, {1, 8}, {2, 8}, {3, 8}, {4, 8}, {4, 7}, {5, 7}, {6, 7}, {1, 7}, {7, 8}}}, {FormatType -> TraditionalForm, FormatType -> TraditionalForm, GraphLayout -> {"Dimension" -> 2}, VertexLabels -> {"Name"}}],#][[1]]/.Association->List))&/@dtsnoniso]


sidewaysbinaryrulegenerate[fg_]:=Module[{dts,newfgs},
dts=detectnonisohexagonwithtwovertsinmiddle[fg];
(* Delete cases with a numerator between 4 and 1 *)
dts=Select[dts,0=!=(fg/.{x[#[[1]],#[[4]]]:>0,x[#[[4]],#[[1]]]:>0})&];
newfgs=If[dts==={},{},(fg*(Numerator[fg]/.x[a_,b_]:>x[a/.{#[[7]]->#[[8]],#[[8]]->#[[7]]},b/.{#[[7]]->#[[8]],#[[8]]->#[[7]]}])/Numerator[fg]/.x[a_,b_]:>x[b,a]/;(a>b))&/@dts];
canonicalizefgraph/@newfgs]


(* ::Section:: *)
(*L+n loop graphs from L loops (only rung rule)*)


generateRungWithCoeff[graphsWithCoeff_]:=Module[{rrgen},
rrgen={#[[1]],fullRungRule[#[[2]]]}&/@graphsWithCoeff;
DeleteDuplicates[Sequence@@Thread[#]&/@rrgen]
]


parallelGenerateRungWithCoeff[graphsWithCoeff_]:=Module[{rrgen},
rrgen=ParallelMap[Sequence@@Thread[{#[[1]],fullRungRule[#[[2]]]}]&,graphsWithCoeff];
DeleteDuplicates[rrgen]
]


(*Gives the edges of a graph as String in the notation of the phyton library networkx, that is a list of tuples*)
edgeListNX[edges_List]:=StringReplace[StringReplace[ StringReplace[ToString[edges/. UndirectedEdge->List],{"{"->"(","}"->")"}],{"(("->"[(","))"->")]"}],"()"->"[]"]


nn=7;


(* ::Subsubsection:: *)
(*n+1 loop from n loops*)


gWithCoeff=Thread[{amplitudeCoefficients[nn],fGraphListcan[nn]}];


result=parallelGenerateRungWithCoeff[gWithCoeff];


data={List@@@(List@@Denominator[#[[2]]]),#[[1]]}&/@result;


sameDen=GatherBy[data,First];


coefDen=Map[Boole[Or@@#]&,Map[!(#[[2]]===0)&,sameDen,{2}]];


denEdges=edgeListNX/@(#[[1,1]]&/@sameDen);


dataDen =Transpose[{coefDen,denEdges}];


csv=Prepend[dataDen,{"COEFFICIENTS","EDGES"}];
Export["den_graph_data_"<>ToString[nn]<>"to"<>ToString[nn+1]<>".csv",csv]


Print[ToString[nn]<>"to"<>ToString[nn+1]<>"Completed. Lenght ", Length[coefDen] ]


(*Test that the result is correct*)
(*Position[fGraphListcan[7],#[[2]]]&/@result;
Flatten[Extract[amplitudeCoefficients[7],%]]
First/@result===%*)


(* ::Subsubsection:: *)
(*n+2 loop from n loops*)


result=parallelGenerateRungWithCoeff[result];


data={List@@@(List@@Denominator[#[[2]]]),#[[1]]}&/@result;


sameDen=GatherBy[data,First];


coefDen=Map[Boole[Or@@#]&,Map[!(#[[2]]===0)&,sameDen,{2}]];


denEdges=edgeListNX/@(#[[1,1]]&/@sameDen);


dataDen =Transpose[{coefDen,denEdges}];


csv=Prepend[dataDen,{"COEFFICIENTS","EDGES"}];
Export["den_graph_data_"<>ToString[nn]<>"to"<>ToString[nn+2]<>".csv",csv]


Print[ToString[nn]<>"to"<>ToString[nn+2]<>"Completed. Lenght ", Length[coefDen] ]


(* ::Subsubsection:: *)
(*n+3 loop from n loops*)


result=parallelGenerateRungWithCoeff[result];


data={List@@@(List@@Denominator[#[[2]]]),#[[1]]}&/@result;


sameDen=GatherBy[data,First];


coefDen=Map[Boole[Or@@#]&,Map[!(#[[2]]===0)&,sameDen,{2}]];


denEdges=edgeListNX/@(#[[1,1]]&/@sameDen);


dataDen =Transpose[{coefDen,denEdges}];


csv=Prepend[dataDen,{"COEFFICIENTS","EDGES"}];
Export["den_graph_data_"<>ToString[nn]<>"to"<>ToString[nn+3]<>".csv",csv]


Print[ToString[nn]<>"to"<>ToString[nn+3]<>"Completed. Lenght ", Length[coefDen] ]


(*Check runge rule is implemented correctly

Position[fGraphListcan[7],#[[2]]]&/@result;
Flatten[Extract[amplitudeCoefficients[7],%]]===First/@result
*)
