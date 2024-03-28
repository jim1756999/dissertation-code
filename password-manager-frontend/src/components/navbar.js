
export default function NavBar({routeIndex, setRouteIndex}){

    const routes = [
        { title: 'Vault', route: '/' },
        { title: 'Checker', route: '/' },
        { title: 'Generator', route: '/' },
    ]

    const listItems = routes.map((item, idx) => 
        <li key={idx} 
        className={`nav-item`}
            onClick={()=>setRouteIndex(idx)}
        >
            <span className={`nav-link ${routeIndex === idx ? 'active' : ''}`}>{item.title}</span>
        </li>
    )

    return (
        
        <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container-fluid">
        <span class="navbar-brand" href="#">Password Manager</span>

          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="nav nav-pills ms-auto">{listItems}</ul>
          </div>
        </div>
      </nav>
    )
}