
const TimeComparison = ({ item }) => {
  // 计算当前时间和item.timestamp之间的差值(ms)
  const currentTime = Date.now(); // 获取当前时间的时间戳
  const timeDifference = Number(currentTime) / 1000 - Number(item); // 计算差值

  // 将时间差转换为天数
  const daysDifference = timeDifference / (1000 * 60 * 60 * 24);

  // If the difference is greater than 180 days, display "Expired"
  return daysDifference > 180 ? <span className="btn btn-outline-danger">Expired</span> : null;
};

export default TimeComparison;