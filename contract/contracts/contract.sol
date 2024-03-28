// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Database {
    struct Data {
        uint id; // 标识
		string name;
        string account; // 账号
        string password; // 密码
        string note; // 备注
        uint timestamp; // 时间戳
        bool isValid; // 该行数据是否有效
    }

    Data[] public datas;
    uint public nextId = 1;

    // 增加数据
    function create(string memory name, string memory account, string memory password, string memory note) public {
        datas.push(Data(nextId, name, account, password, note, block.timestamp, true));
        nextId++;
    }

    // 查找数据索引
    function find(uint id) internal view returns(uint) {
        for(uint i = 0; i < datas.length; i++) {
            if(datas[i].id == id) {
                return i;
            }
        }
        revert('Data not found.');
    }

    // 读取数据
    function read(uint id) public view returns(uint, string memory, string memory, string memory, string memory, uint, bool) {
        uint i = find(id);
        return(datas[i].id, datas[i].name, datas[i].account, datas[i].password, datas[i].note, datas[i].timestamp , datas[i].isValid);
    }

    // 更新数据
    function update(uint id, string memory name, string memory account, string memory password, string memory note, bool isValid) public {
        uint i = find(id);
        datas[i].name = name;
        datas[i].account = account;
        datas[i].password = password;
        datas[i].note = note;
        datas[i].timestamp = block.timestamp;
        datas[i].isValid = isValid;
    }

    // 删除数据
    function deleteData(uint id) public {
        uint i = find(id);
        datas[i].name = "0";
        datas[i].account = "0";
        datas[i].password = "0";
        datas[i].note = "0";
        datas[i].timestamp = block.timestamp;
        datas[i].isValid = false;
    }
}