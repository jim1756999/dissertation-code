import { useState } from 'react'
import { newRecord, updateRecord } from '../utils/c'

export default function NewEntryDialog({onConfirm, setDialogVisible, form, input, dialogType}){

    function handleConfirm(){
        if(dialogType === 'add'){
            newRecord(form.name, form.account, form.password, form.note).then(res => {
                console.log(res, 'newRecord Success')
                setTimeout(() => {
                    onConfirm()
                }, 500);
            })
        }else if(dialogType === 'edit'){
            updateRecord(form.id, form.name, form.account, form.password, form.note, 1).then(res => {
                console.log(res, 'updateRecord Success')
                setTimeout(() => {
                    onConfirm()
                }, 500);
            })
        }else if(dialogType === 'view'){
            setDialogVisible(false)
        }
    }
    return (
        <div className="dialog">
            <div className="dialog-content" onClick={e=>e.stopPropagation()}>
                <div><input className="form-control" placeholder="Website" disabled={dialogType==='view'} value={form.name} onChange={e => input('name', e.target.value)}/></div>
                <div><input className="form-control" placeholder="Account" disabled={dialogType==='view'} value={form.account} onChange={e => input('account', e.target.value)}/></div>
                <div><input className="form-control" placeholder="Password" disabled={dialogType==='view'} value={form.password} onChange={e => input('password', e.target.value)}/></div>
                <div><input className="form-control" placeholder="Note" disabled={dialogType==='view'} value={form.note} onChange={e => input('note', e.target.value)}/></div>
                <button className="btn btn-primary"  disabled={dialogType==='view'} onClick={handleConfirm}>Confirm</button>
                <button className="btn btn-danger" onClick={()=>setDialogVisible(false)}>Cancel</button>
            </div>
        </div>

    )
}