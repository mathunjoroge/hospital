<?php
     session_start();
     include "../connect.php";
    $date=date('Y-m-d');
    $fees = $_POST["fees"];
$List = implode(", ", $fees);
$result = $db->prepare(
    "SELECT GROUP_CONCAT(amount) AS amount FROM  fees   WHERE fees_id IN($List)"
);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {
    $feeamount = $row["amount"];

    $amounts = explode(",", $feeamount);

    foreach (array_combine($fees, $amounts) as $fee => $amount) {
        $token = $_POST["token"];
        $c = $_POST["patient"];
        $sql =
            "INSERT INTO collection (fees_id,date,paid_by,amount,token) VALUES (:a,:b,:c,:d,:token)";
        $q = $db->prepare($sql);
        $q->execute([
            ":a" => $fee,
            ":b" => $date,
            ":c" => $c,
            ":d" => $amount,
            ":token" => $token
        ]);
    }
}
   //set patient available at cashier
   $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$c));  
      if (isset($_POST['pro'])) {
     header("location:pro.php?response=1&token=$token&resp=1&search=$c");
 } 
 else{
     header("location:procedure.php?response=1&token=$token&resp=1&search=$c");
 }
?>
      