import { useState } from "react"

export default function PasswordChecker() {
    const [password, setPassword] = useState('')
    const [text, setText] = useState('')

    const [type, setType] = useState(undefined)

    function handleChecker() {
        console.log('handleChecker')
        fetch('http://localhost:5000/checker', {
            method: 'POST',
            headers: {
                'content-type': 'application/json'
            },
            body: JSON.stringify({
                password: [password]
            })
        })
            .then(res => res.json())
            .then(res => {
                if(res.error){
                    setText(res.error)
                }else{
                    setType(res[0])
                }
                
            })
    }

    return (
<div className="container d-flex">
  <div className="py-4">
    <div className="mb-2">
      <h1 className="display-4">Password Checker</h1>
    </div>
    <div className="row">
      <div className="col">
        <input
          class="form-control mt-2"
          placeholder="Password"
          value={password}
          onChange={e => {
            setPassword(e.target.value);
            handleChecker();
          }}
        />
      </div>
      <div className="container d-flex row mt-3">
      <div className="col">
        <button class="btn btn-primary" onClick={handleChecker}>
          Check
        </button>
        </div>
        </div>
    </div>
    {type !== undefined && (
      <div className="mt-3">
        <div className="password account-list-item">
          <div className={`progress`}>
            <div
              className={`progress-bar ${
                type === 0
                  ? "bg-danger"
                  : type === 1
                  ? "bg-warning"
                  : "bg-success"
              }`}
              role="progressbar"
              style={{ width: `${type === 0 ? "33%" : type === 1 ? "66%" : "100%"}` }}
              aria-valuenow={type === 0 ? "33" : type === 1 ? "66" : "100"}
              aria-valuemin="0"
              aria-valuemax="100"
            ></div>
          </div>
          <div className="text-center mt-2">
            {type === 0 ? 'Weak' : type === 1 ? 'Medium' : 'Strong'}
          </div>
        </div>
      </div>
    )}
  </div>
</div>
    )
}