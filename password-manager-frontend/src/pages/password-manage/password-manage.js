import { useEffect, useState } from "react"
import AccountList from "../../components/account-list"
import NewEntryDialog from "../../components/new-entry-dialog"
import { fetchDataAndCount } from "../../utils/c"


export default function PasswordManage(){
    const [list, setList] = useState([])
    const [dialogVisible, setDialogVisible] = useState(false)
    const [dialogType, setDialogType] = useState('add')
    const [form, setForm] = useState({
        id: '',
        name: '',
        account: '',
        password: '',
        note: '',
        isValid: true
    })

    useEffect(() => {
        getData()
    }, [])

    function getData(){
        fetchDataAndCount().then(res => {
            console.log(res, 'success')
            setList(res)
        })
    }

    function onConfirm(){
        getData()
        setDialogVisible(false)
        resetForm()
        setDialogType('add')
    }

    function onEdit(e){
        console.log(e)
        input('id', e[0])
        input('name', e[1])
        input('account', e[2])
        input('password', e[3])
        input('note', e[4])
        input('isValid', e[6])
        setDialogType('edit')
        setDialogVisible(true)
    }

    function onAdd(){
        resetForm()
        setDialogType('add')
        setDialogVisible(true)
    }

    function onView(e){
        console.log(e)
        input('id', e[0])
        input('name', e[1])
        input('account', e[2])
        input('password', e[3])
        input('note', e[4])
        input('isValid', e[6])
        setDialogType('view')
        setDialogVisible(true)
    }

    function input(field, value) {
        setForm(prevState => ({
            ...prevState,
            [field]: value
        }));
    }

    function resetForm(){
        setForm({
            id: '',
            name: '',
            account: '',
            password: '',
            note: '',
        })
    }

    return (
    <div className="password-manage pubw container">
        <div className="container py-4">
            <div className="row mb-2">
                <div className="col-12">
                    <h1 className="display-4">Vault</h1>
                </div>
                <div className="col-12">
                    <button type="button" class="btn btn-success" onClick={onAdd}>New Entry</button>
                </div>

            </div>
            <div className="row">
                    <div className="col-12">
                        <AccountList list={list} refresh={getData} onEdit={onEdit} onView={onView}/>
                    </div>
                </div>
        </div>
        {dialogVisible ?
        <NewEntryDialog form={form} input={input} onConfirm={onConfirm} setDialogVisible={setDialogVisible} dialogType={dialogType}/>
        : ''
        }
    </div>
    )
}