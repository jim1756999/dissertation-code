import './App.css';
import NavBar from './components/navbar';
import { useState } from 'react'
import PasswordManage from './pages/password-manage/password-manage';
import PasswordGenerator from './pages/password-generator/password-generator';
import PasswordChecker from './pages/password-checker/password-checker';

function App() {
  const [routeIndex, setRouteIndex] = useState(0)

  return (
    <div className="App">
      <NavBar routeIndex={routeIndex} setRouteIndex={setRouteIndex}/>
      { routeIndex === 0 ? <PasswordManage/> : routeIndex === 2 ? <PasswordGenerator/> : routeIndex === 1 ? <PasswordChecker/> : '' }
    </div>
  );
}

export default App;
