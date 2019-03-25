const bincaller = require('./bincaller.js')

drawedgesscript = "drawedges.py";
drawedgesfunctions = ["trippy1","trippy2","directionglow","mosaic","net",
                        "noiser","colorblob","formsurfaceblobs","smallformsurfaceblobs",
                        "smallsurfaceblobs","edges"];

function callFileProcFunctionByName(functionName,srcFile,dstFile){
    if(functionName === "pixelart"){
        return bincaller.regiongrow(srcFile,dstFile);
    }
    else if(drawedgesfunctions.includes(functionName)){
        return bincaller.drawedges(functionName,srcFile,dstFile);
    }
    else{
        console.log("Called function was not registered: " + functionName)
    }
    
    /*switch(functionName){
        case functionName === "pixelart":
            return bincaller.regiongrow(path,destpath);
        case drawedgesfunctions.includes(functionName):
            return bincaller.drawedges(functionName,path,destpath);
        default:
            console.log("Called function was not registered: " + functionName)
    }*/
}

module.exports = {
    'callFileProcFunctionByName':callFileProcFunctionByName
}