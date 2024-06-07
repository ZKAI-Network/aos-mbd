# [WIP] aos-onnx

**aos-onnx** combines the ao operating system module and ONNX Realtime Runtime to create an ao custom module that enables AI inference from *any model* exported as in the ONNX format to your aos experience. 

> The bulk of this effort was done by @samcamwilliams, @twilson63 on [aos-llama](https://github.com/samcamwilliams/aos-llama) & @elliotsayes during the Hack the Weave competition, Elliot was able to create this WASM Binary that includes both SQLite, LUA, and aos, as an `ao` Module.


## AO Resources

* https://ao.arweave.dev
* https://cookbook_ao.arweave.dev

---

This project builds the AOS-SQLITE WASM Binary and Publishes it to Arweave.

## Build Process

1. Build docker image

```sh
cd container
./build.sh
```

2. Get Latest aos module

```sh
git submodule init
git submodule update --remote
```

3. Use docker image to compile process.wasm

```sh
cd aos/process
docker run -v .:/src p3rmaw3b/ao emcc-lua
```

4. Publish Module with tags via arkb

> You will need a funded wallet for this step 

```sh
export WALLET=~/.wallet.json
npm run deploy
```
