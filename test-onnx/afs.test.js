const { describe, it } = require('node:test')
const assert = require('assert')
const weaveDrive = require('./weavedrive.js')
const fs = require('fs')
const wasm = fs.readFileSync('./AOS.wasm')
const m = require(__dirname + '/AOS.js')
const AdmissableList =
  [
    "dx3GrOQPV5Mwc1c-4HTsyq0s1TNugMf7XfIKJkyVQt8", // Random NFT metadata (1.7kb of JSON)
    "XOJ8FBxa6sGLwChnxhF2L71WkKLSKq1aU5Yn5WnFLrY", // GPT-2 117M model.
    "M-OzkyjxWhSvWYF87p0kvmkuAEEkvOzIj4nMNoSIydc", // GPT-2-XL 4-bit quantized model.
    "kd34P4974oqZf2Db-hFTUiCipsU6CzbR6t-iJoQhKIo", // Phi-2
    "ISrbGzQot05rs_HKC08O_SmkipYQnqgB1yC3mjZZeEo", // Phi-3 Mini 4k Instruct
    "sKqjvBbhqKvgzZT4ojP1FNvt4r_30cqjuIIQIr-3088", // CodeQwen 1.5 7B Chat q3
    "Pr2YVrxd7VwNdg6ekC0NXWNKXxJbfTlHhhlrKbAd1dA", // Llama3 8B Instruct q4
    "jbx-H6aq7b3BbNCHlK50Jz9L-6pz9qmldrYXMwjqQVI"  // Llama3 8B Instruct q8
  ]

describe('AOS-ONNX Tests', async () => {
  var instance;
  const handle = async function (msg, env) {
    console.log(instance.cwrap)
    const res = await instance.cwrap('handle', 'string', ['string', 'string'], { async: true })(JSON.stringify(msg), JSON.stringify(env))
    console.log('Memory used:', instance.HEAP8.length)
    return JSON.parse(res)
  }

  it('Create instance', async () => {
    console.log("Creating instance...")
    var instantiateWasm = function (imports, cb) {

      // merge imports argument
      const customImports = {
        env: {
          memory: new WebAssembly.Memory({ initial: 8589934592 / 65536, maximum: 17179869184 / 65536, index: 'i64' })
        }
      }
      //imports.env = Object.assign({}, imports.env, customImports.env)

      WebAssembly.instantiate(wasm, imports).then(result =>

        cb(result.instance)
      )
      return {}
    }

    instance = await m({
      admissableList: AdmissableList,
      WeaveDrive: weaveDrive,
      ARWEAVE: 'https://arweave.net',
      mode: "test",
      blockHeight: 100,
      spawn: {
        "Scheduler": "TEST_SCHED_ADDR"
      },
      process: {
        id: "TEST_PROCESS_ID",
        owner: "TEST_PROCESS_OWNER",
        tags: [
          { name: "Extension", value: "Weave-Drive" }
        ]
      },
      instantiateWasm
    })
    await new Promise((r) => setTimeout(r, 1000));
    console.log("Instance created.")
    await new Promise((r) => setTimeout(r, 250));

    assert.ok(instance)
  })

  it('Eval Lua', async () => {
    console.log("Running eval")
    const result = await handle(getEval('1 + 1'), getEnv())
    console.log("Eval complete")
    assert.equal(result.response.Output.data.output, 2)
  })

  it('Add data to the VFS', async () => {
    await instance['FS_createPath']('/', 'data')
    const onnxData = await fs.promises.readFile('model.onnx');
    await instance['FS_createDataFile']('/', 'data/model.onnx',onnxData, true, false, false)
    const result = await handle(getEval('return "OK"'), getEnv())
    assert.ok(result.response.Output.data.output == "OK")
  })

  function arrayToList(array) {
      let list = [];
      for (let i = array.length - 1; i >= 0; i--) {
          list[i] = array[i];
      }
      console.log(list);
      return list;
  }


  it('Llama Lua library loads', async () => {
    const result = await handle(getEval(`
local Ort = require "ort"
local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("/data/model.onnx", SessionOptions)
local tensorA = Ort.CreateValue({ 3, 4 }, "FLOAT", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
local tensorB = Ort.CreateValue({ 4, 3 }, "FLOAT", {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120})
local result = Session:Run {
a = tensorA,
b = tensorB
}
local dataC = result.c:GetData()
return pairs(dataC)
`), getEnv())
    expected_output = [
   700,  800,  900,
  1580, 1840, 2100,
  2460, 2880, 3300
];
    // Not sure why the output is called Error. Seems like an issue to handle later for sure. 
    assert.equal(JSON.stringify(arrayToList(result.response.Error)),JSON.stringify(expected_output));
  })

  it.skip('AOS runs Simple ONNX file', async () => {
    const result =
      await handle(
        getEval(fs.readFileSync("onnx-test.lua", "utf-8")),
        getEnv()
      )
    console.log(result)
    assert.ok(result.response.Output.data.output.includes("<|im_end|>"))
  })

})

function getEval(expr) {
  return {
    Id: '1',
    Owner: 'TOM',
    Module: 'FOO',
    From: 'foo',
    'Block-Height': '1000',
    Timestamp: Date.now(),
    Tags: [
      { name: 'Action', value: 'Eval' }
    ],
    Data: expr
  }
}

function getEnv() {
  return {
    Process: {
      Id: 'AOS',
      Owner: 'TOM',
      Tags: [
        { name: 'Name', value: 'Thomas' }
      ]
    }
  }
}
