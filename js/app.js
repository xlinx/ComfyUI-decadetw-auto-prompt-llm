// //ref: https://note.com/nyaoki_board/n/na7c54c9ae2a5
//
// import { app } from "/scripts/app.js";
//
// app.registerExtension({
// 	name: "Comfy.🧩 Auto-Prompt-LLM.Auto-LLM-Text-Vision",
//     async beforeRegisterNodeDef(nodeType, nodeData, app) {
//         if (nodeData.name === "Batch String") {
//             nodeType.prototype.onNodeCreated = function () {
//                 this.getExtraMenuOptions = function(_, options) {
//                     options.unshift(
//                         {
//                             content: "add input",
//                             callback: () => {
//                                 var index = 1;
//                                 if (this.inputs != undefined){
//                                     index += this.inputs.length;
//                                 }
//                                 this.addInput("text" + index, "STRING", {"multiline": true});
//                             },
//                         },
//                         {
//                             content: "remove input",
//                             callback: () => {
//                                 if (this.inputs != undefined){
//                                     this.removeInput(this.inputs.length - 1);
//                                 }
//                             },
//                         },
//                     );
//                 }
//             }
//         }
//     },
// 	loadedGraphNode(node, _) {
// 		if (node.type === "Batch String") {
// 		}
// 	},
// });