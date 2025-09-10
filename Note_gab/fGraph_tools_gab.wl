(* ::Package:: *)

decoratedGraphs[fGraphExpression_]/;!Head[fGraphExpression]===List:=Module[{parsedEdges=DeleteCases[({#1[[1,1]],Total[#1[[All,2]]]}&)/@GatherBy[fGraphExpression/. Times->List/. {x[q__]^z_:>{Sort[{q}],-z},x[q__]:>{Sort[{q}],-1}},First],{x_,0}],multi},multi=If[Length[Select[parsedEdges,Last[#1]>1&]]==0,Join[Select[parsedEdges,Last[#1]===1&],({#1,Abs[#2]+1}&)@@@Select[parsedEdges,Last[#1]<0&]],Join[Select[parsedEdges,Last[#1]===1&],({#1,2 Abs[#2]}&)@@@Select[parsedEdges,Last[#1]<0&],({#1,2 Abs[#2]-1}&)@@@Select[parsedEdges,Last[#1]>1&]]];Graph[UndirectedEdge@@@Join@@Function[{no},If[Last[multi[[no]]]===1,{First[multi[[no]]]},Partition[RotateLeft[Join[Reverse[multi[[no,1]]],(e[no,#1]&)/@Range[multi[[no,2]]]]],2,1]]]/@Range[Length[multi]]]]

decoratedGraphs[{fGraphExpressions__}]:=decoratedGraphs/@{fGraphExpressions}


fromDecoratedToIntegrand[graph_]:= Module[{edgeList,sources,sinks,dashedEdges},
edgeList=EdgeList[graph];
sources=Cases[edgeList, UndirectedEdge[a_,e[i_,j_]]/;AtomQ[a]];
sinks=Cases[edgeList, UndirectedEdge[e[i_,j_],a_]/;AtomQ[a]];
dashedEdges=Table[Sequence@@ConstantArray[Sort[sources[[i,1]]\[UndirectedEdge]sinks[[i,2]]],sinks[[i,1,2]]-1] ,{i,Length[sources]}];
(Times@@(x@@@dashedEdges))/Times@@(x@@@Cases[edgeList,a_\[UndirectedEdge]b_/;And[AtomQ[a],AtomQ[b]]])
]


canonicalFGraph[graph_]:=fromDecoratedToIntegrand@CanonicalGraph[ decoratedGraphs[graph]]
