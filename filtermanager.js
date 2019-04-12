const bincaller = require('./bincaller.js')
const utilfns = require('./utilfns')
const fs = require('fs')
const util = require('util')
const copyfile = util.promisify(fs.copyFile);

drawedgesscript = "drawedges.py";
drawedgesfunctions = ["trippy1","trippy2","directionglow","mosaic","net",
                        "noiser","colorblob","formsurfaceblobs","smallformsurfaceblobs",
                        "smallsurfaceblobs","pencil"];

function compose(functionNames,srcFile,dstFile){
    
    return new Promise((resolve,reject)=>{

        if(Array.isArray(functionNames)){

            var filePartPromises = []
            var cnt = 0
            functionNames.forEach((fnName)=>{
                var partialDstFile = utilfns.getNewPath(srcFile,"-part"+ cnt);
                console.log("Path destiniy ;): " + partialDstFile)
                filePartPromises.push(callFileProcFunctionByName(fnName,srcFile,partialDstFile));
                cnt++;
            });

            Promise.all(filePartPromises)
            .then((destinationImgs)=>{
                console.log("Promises fulfilled successfully")
                //console.log("Promises fulfilled successfully: " + destinationImgs)
                return bincaller.compose("default",destinationImgs,dstFile);
            })
            .then((finalImg)=>{
                return resolve(finalImg);
            })
            .catch((err)=>{
                return reject("Error occured when composing fn: " + err);
            });
        }
        else{
            callFileProcFunctionByName(functionNames,srcFile,dstFile)
            .then((finalImg)=>{
                return resolve(finalImg);
            })
            .catch((err)=>{
                return reject("Error occured when composing single fn: " + err);
            });
        }
    });
    
}

function callFileProcFunctionByName(functionName,srcFile,dstFile){
    if(functionName === "pixelart"){
        return bincaller.regiongrow(srcFile,dstFile,200000);
    }
    if(functionName === "regiongrowf"){
        return bincaller.regiongrow(srcFile,dstFile,1000000);
    }
    else if(drawedgesfunctions.includes(functionName)){
        return bincaller.drawedges(functionName,srcFile,dstFile);
    }
    else if(functionName === "pass"){
        return copyImage(srcFile,dstFile)
    }
    else{
        console.log("Called function was not registered: " + functionName)
    }
}

function copyImage(srcFile,dstFile){
    return new Promise((resolve,reject)=>{
        copyfile(srcFile,dstFile)
        .then(()=>{
            return resolve(dstFile);
        })
        .catch((err)=>{
            return reject(err);
        })
    });
}

module.exports = {
    'callFileProcFunctionByName':callFileProcFunctionByName,
    'compose':compose
}