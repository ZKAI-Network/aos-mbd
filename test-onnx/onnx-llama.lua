local Ort = require "ort"
local png = require "luapng"
local vec = require "vec"

print("Reading girl.png")
local imagedata = png.read("girl.png")
local imagetensor = Ort.CreateValue({ 1, 3, imagedata.height, imagedata.width }, "FLOAT",  vec.div(imagedata, 255))

print("Reading mask.png")
local maskdata = png.read_grayscale("mask.png")
local masktensor = Ort.CreateValue({ 1, 1, maskdata.height, maskdata.width }, "FLOAT",  vec.div(maskdata, 255))

local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
print "Loading model"
local Session = Env:CreateSession("lama_fp32.onnx", SessionOptions)

print "Run"
local outputvalues = Session:Run {
	image = imagetensor,
	mask = masktensor,
}

local outputImage = outputvalues.output:GetData()
outputImage.height, outputImage.width = imagedata.height, imagedata.width
png.write(outputImage, "out.png")
print "finish"