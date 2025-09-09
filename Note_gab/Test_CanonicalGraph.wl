(* ::Package:: *)

(* ::Input:: *)
(*decoratedGraphs[fGraphExpression_]/;!Head[fGraphExpression]===List:=Module[{parsedEdges=DeleteCases[({#1[[1,1]],Total[#1[[All,2]]]}&)/@GatherBy[fGraphExpression/. Times->List/. {x[q__]^z_:>{Sort[{q}],-z},x[q__]:>{Sort[{q}],-1}},First],{x_,0}],multi},multi=If[Length[Select[parsedEdges,Last[#1]>1&]]==0,Join[Select[parsedEdges,Last[#1]===1&],({#1,Abs[#2]+1}&)@@@Select[parsedEdges,Last[#1]<0&]],Join[Select[parsedEdges,Last[#1]===1&],({#1,2 Abs[#2]}&)@@@Select[parsedEdges,Last[#1]<0&],({#1,2 Abs[#2]-1}&)@@@Select[parsedEdges,Last[#1]>1&]]];Graph[UndirectedEdge@@@Join@@Function[{no},If[Last[multi[[no]]]===1,{First[multi[[no]]]},Partition[RotateLeft[Join[Reverse[multi[[no,1]]],(e[no,#1]&)/@Range[multi[[no,2]]]]],2,1]]]/@Range[Length[multi]]]]*)
(**)
(*decoratedGraphs[{fGraphExpressions__}]:=decoratedGraphs/@{fGraphExpressions}*)


(* ::Input:: *)
(*decoratedGraphs[x[1,2]/x[2,3]]*)
(*EdgeList@%*)


(* ::Input:: *)
(*g={"a"\[UndirectedEdge]"c","a"\[UndirectedEdge]"f","a"\[UndirectedEdge]"h","a"\[UndirectedEdge]"g","a"\[UndirectedEdge]"e","a"\[UndirectedEdge]"i","b"\[UndirectedEdge]"d","b"\[UndirectedEdge]"e","b"\[UndirectedEdge]"g","b"\[UndirectedEdge]"h","b"\[UndirectedEdge]"f","b"\[UndirectedEdge]"j","c"\[UndirectedEdge]"i","c"\[UndirectedEdge]"d","c"\[UndirectedEdge]"j","c"\[UndirectedEdge]"f","d"\[UndirectedEdge]"j","d"\[UndirectedEdge]"i","d"\[UndirectedEdge]"e","e"\[UndirectedEdge]"g","e"\[UndirectedEdge]"i","f"\[UndirectedEdge]"j","f"\[UndirectedEdge]"h","g"\[UndirectedEdge]"h","a"\[UndirectedEdge]e[25,1],e[25,1]\[UndirectedEdge]e[25,2],e[25,2]\[UndirectedEdge]"b","a"\[UndirectedEdge]e[26,1],e[26,1]\[UndirectedEdge]e[26,2],e[26,2]\[UndirectedEdge]"d","b"\[UndirectedEdge]e[27,1],e[27,1]\[UndirectedEdge]e[27,2],e[27,2]\[UndirectedEdge]"c","e"\[UndirectedEdge]e[28,1],e[28,1]\[UndirectedEdge]e[28,2],e[28,2]\[UndirectedEdge]"f"};*)


(* ::Input:: *)
(*Print[Graph[g]===CanonicalGraph[g]]*)
