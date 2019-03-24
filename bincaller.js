
const fs = require('fs');
const spawnChild = require("child_process").spawn;
const util = require('util')
const fstatpromise = util.promisify(fs.lstat);

const binarydirectory = __dirname + "/binaries/";
var binarylocations = {};
var binarynames = ["RegionGrowing"];

binarynames.forEach((binaryname)=>{
    var binarylocation = "";
    if(process.platform === "win32"){
        binarylocation = binarydirectory + binaryname + ".exe";
    }
    else{
        binarylocation = binarydirectory + binaryname;
    }
    
    if(!fs.lstatSync(binarylocation).isFile()){
        console.log("Location: " + binarylocation + " does not exist - exiting");
        process.exit(4);
    }
    binarylocations[binaryname] = binarylocation
});

function handlefileprocess(childprocess,destFile){
    return new Promise((resolve,reject)=>{
        
        childprocess.stdout.on('data', (data) => {
            if(logging){
                console.log(data.toString('utf8'));
            }
        });
        childprocess.stderr.on('data', (data) => {
            if(logging){
                console.log(data.toString('utf8'));
            }
        });
        childprocess.stdout.on('close', (code) => {
            //fast cast to int
            code = code | 0;
            console.log("Program closed: " + code);
            if(code === 0){
                console.log("binary: binaryloc execution fininshed")
                fstatpromise(destFile)
                .then((stat)=>{
                    if(stat.isFile()){
                        return resolve(destFile);
                    }
                })
                .catch((err)=>{
                    return reject("ERROR: destination file " + destFile + " does not seem to exist");
                })
            }
            else{
                return reject("ERROR: Program exitted with code " + code);
            }
        });
    });
}

var logging = true;
function execfileprocbin(binaryloc,sourceFile,destFile){
    //const execcommand = `${binaryloc} "${sourceFile}" "${destFile}"`;
    //console.log("Executing: " + execcommand);
    //const childprocess = spawnChild(execcommand);
    const childprocess = spawnChild(binaryloc,[ sourceFile, destFile ]);
    return handlefileprocess(childprocess,destFile);
}

function execfileprocpyscript(scriptlocation,sourceFile,destFile){
    exectype = "python3";
    if(process.platform === "win32"){
        exectype = "python"
    }
    const childprocess = spawnChild(exectype,[scriptlocation, sourceFile, destFile ]);
    return handlefileprocess(childprocess,destFile);
}

var regiongrowbin = binarylocations["RegionGrowing"]
if(!regiongrowbin){
    console.log("regiongrowbin not in locations");
    exit(4);
}
function regiongrow(sourceFile,destFile){
    console.log("Calling regiongrow: " + sourceFile + " / " + destFile);
    return execfileprocbin(regiongrowbin,sourceFile,destFile);
}

function cartoonifyme(sourceFile,destFile){
    console.log("Calling cartoonifyme: " + sourceFile + " / " + destFile);
    return execfileprocpyscript("cartoonifyme.py",sourceFile,destFile);
}

module.exports = {
    'regiongrow': regiongrow,
    'cartoonifyme': cartoonifyme
}