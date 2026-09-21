function QuickQuestion({ title, detail, color, onChoose }) {
  return (
    <button
      type="button"
      className={`quick-question quick-question-${color}`}
      onClick={onChoose}
    >
      <strong>{title}</strong>
      <span>{detail}</span>
    </button>
  )
}

export default QuickQuestion
