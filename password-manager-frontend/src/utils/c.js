const { contractABI, contractAddress, data } = require('../config.js')
const {Web3} = require('web3');
const web3 = new Web3('http://localhost:7545');

const myContract = new web3.eth.Contract(contractABI, contractAddress);

export async function newRecord(name, account, password, note) {
    console.log(name, account, password, note, 'name, account, password, note')
    myContract.methods.create(name, account, password, note).send(data)
    .then(function(receipt){
        console.log(receipt);
    })
    .catch(function(error) {
        console.error('Data creation failed:', error);
    });
}

export async function deleteData(id) {
  console.log(id)
    try {
        const result = await myContract.methods.deleteData(id).send(data);
        console.log('Data deletion successful:', result);
    } catch (error) {
        console.error('Data deletion failed:', error);
    }
}

export async function updateRecord(id, name, account, password, note, isValid) {
  console.log(id, name, account, password, note, isValid, 'id, name, account, password, note, isValid')
    try {
        const result = await myContract.methods.update(id, name, account, password, note, isValid).send(data);
        console.log('Record update successful:', result);
    } catch (error) {
        console.error('Record update failed:', error);
    }
}

async function readData(id) {
    try {
        const result = await myContract.methods.read(web3.utils.toBigInt(id)).call();
        console.log('Data:', result);
        return result;
    } catch (error) {
        console.error('Error reading data:', error);
        throw error;
    }
}

async function fetchTotalCount() {
    try {
        const nextId = await myContract.methods.nextId().call();
        const totalCount = web3.utils.toNumber(nextId) - 1;
        console.log('Total count:', totalCount);
        return totalCount;
    } catch (error) {
        console.error('Error fetching total count:', error);
        throw error;
    }
}

export async function fetchDataAndCount() {
    const totalCount = await fetchTotalCount();
    const data = [];
    for(let i = 0; i < totalCount; i++) {
        const record = await readData(i+1);
        if(!record[6]) continue;
        data.push(record);
    }

    return data;
}
