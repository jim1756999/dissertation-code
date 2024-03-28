import { deleteData } from "../utils/c"
import "./if_expiry"
import TimeComparison from "./if_expiry"

export default function AccountList({list, refresh, onEdit, onView}){
    console.log('AccountList')

    function handleDelete(id){
        deleteData(id).then(() => {
            refresh()
        })
    }

    const listItems = list.map((item, idx) => 
        <li className="account-list-item list-group-item d-flex justify-content-between align-items-center" key={idx}>
            <div>
                <h3 className="title">{item[1]}</h3>
                <p className="name">{item[2]}</p>
            </div>
            <div className="options">
                    <span><TimeComparison item={item[5]}/> </span>
                <div className="btn-group">
                <button type="button" onClick={()=>onView(item)} className="btn btn-primary">View</button>
                <button type="button" onClick={()=>onEdit(item)} className="btn btn-warning">Edit</button>
                <button type="button" className="btn btn-danger" onClick={()=>handleDelete(item[0])} >Delete</button>
                </div>


            </div>
        </li>    
    )

    return (
        <ul className="account-list list-group">
            {listItems}
        </ul>
    )
}